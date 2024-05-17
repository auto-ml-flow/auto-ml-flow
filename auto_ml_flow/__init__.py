import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from loguru import logger

from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.client.v1.consts import Status
from auto_ml_flow.client.v1.models.experiments import ExperimentModel
from auto_ml_flow.client.v1.models.runs import RunModel
from auto_ml_flow.client.v1.models.systems import CreateSystemPayload
from auto_ml_flow.handlers.experiment import get_or_create_experiment
from auto_ml_flow.handlers.run import run_ended, run_started
from auto_ml_flow.handlers.run_metric import add_metric_to
from auto_ml_flow.handlers.system import create_system
from auto_ml_flow.metrics.monitor import SystemMetricsMonitor
from auto_ml_flow.metrics.system import get_system


class AutoMLFlow:
    _client: AutoMLFlowClient | None = None
    _latest_run: RunModel | None = None
    _experiment: ExperimentModel | None = None

    @classmethod
    def set_tracking_url(cls, url: str) -> None:
        cls._client = AutoMLFlowClient(base_url=url)

    @classmethod
    def start_experiment(cls, name: str) -> None:
        if cls._client is None:
            raise ValueError("Tracking URL is not set. Use 'set_tracking_url' method to set it.")

        cls._experiment = get_or_create_experiment(name, cls._client)

    @classmethod
    @contextmanager
    def start_run(cls, description: str) -> Generator[Any, Any, None]:
        if cls._client is None:
            raise ValueError("Tracking URL is not set. Use 'set_tracking_url' method to set it.")

        if cls._experiment is None:
            raise ValueError(
                "Experiment not started."
                "Start experiment with 'AutoMLFlow.start_experiment(\"My experiment for test\")'"
            )

        system_info = get_system()  # before run creation because it pretty hard task
        run = run_started(
            client=cls._client, experiment_id=cls._experiment.id, description=description
        )
        start_time = datetime.now()

        logger.debug(
            f"New {run.id=} for experiment {cls._experiment.name} (id={cls._experiment.id}) created"
        )

        cls._latest_run = run

        system = create_system(
            CreateSystemPayload(run=run.id, **system_info.model_dump()), cls._client
        )
        monitor = SystemMetricsMonitor(system=system.id, client=cls._client, interval=0.5)
        monitor.start()

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
        finally:
            monitor.finish()

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
