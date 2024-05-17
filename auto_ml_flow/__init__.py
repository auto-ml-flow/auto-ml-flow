import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from loguru import logger

from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.client.v1.consts import Status
from auto_ml_flow.client.v1.models.experiments import ExperimentModel
from auto_ml_flow.client.v1.models.runs import RunModel
from auto_ml_flow.handlers.experiment import create_experiment, get_experiment_by
from auto_ml_flow.handlers.run import run_ended, run_started
from auto_ml_flow.handlers.run_metric import add_metric_to


class AutoMLFlow:
    _client: AutoMLFlowClient | None = None
    _latest_run: RunModel | None = None

    @classmethod
    def set_tracking_url(cls, url: str) -> None:
        cls._client = AutoMLFlowClient(base_url=url)

    @classmethod
    @contextmanager
    def experiment_manager(cls, name: str) -> Generator[ExperimentModel, Any, None]:
        if cls._client is None:
            raise ValueError("Tracking URL is not set. Use 'set_tracking_url' method to set it.")

        experiment = None
        try:
            name = name or str(uuid.uuid4())

            experiment = get_experiment_by(name, cls._client)

            yield experiment
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Experiment retrieval failed: {e}")

            yield create_experiment(name, cls._client)

    @classmethod
    @contextmanager
    def run_manager(
        cls, experiment: ExperimentModel, description: str
    ) -> Generator[Any, Any, None]:
        if cls._client is None:
            raise ValueError("Tracking URL is not set. Use 'set_tracking_url' method to set it.")

        run = run_started(client=cls._client, experiment_id=experiment.id, description=description)
        start_time = datetime.now()
        logger.debug(f"New {run.id=} for experiment {experiment.name} (id={experiment.id}) created")

        cls._latest_run = run

        try:
            yield run
            end_time = datetime.now()
            duration = end_time - start_time
            logger.debug(f"{run.id=} ended with {duration=}")
            run_ended(
                client=cls._client,
                run_id=run.id,
                status=Status.DONE,
                duration=duration.total_seconds(),
            )
        except Exception:
            end_time = datetime.now()
            duration = end_time - start_time
            error_trace = traceback.format_exc()
            logger.debug(error_trace)
            logger.debug(f"{run.id=} ended with {duration=}")
            run_ended(
                client=cls._client,
                run_id=run.id,
                status=Status.FAILED,
                duration=duration.total_seconds(),
                traceback=error_trace,
            )
            raise

    @classmethod
    def log_metric(cls, key: str, value: float) -> None:
        if cls._client is None:
            raise ValueError("Tracking URL is not set. Use 'set_tracking_url' method to set it.")

        if not cls._latest_run:
            raise ValueError(
                "Not found current run. "
                "First need to call 'with AutoMLFlow.run_manager(experiment)'"
            )

        add_metric_to(cls._latest_run, key, value, cls._client)
