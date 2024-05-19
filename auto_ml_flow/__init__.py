import pickle
import tempfile
import traceback
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from typing import IO, Any, Generator

import numpy as np
import pandas as pd
from loguru import logger

from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.client.v1.consts import Status
from auto_ml_flow.client.v1.models.experiments import ExperimentModel
from auto_ml_flow.client.v1.models.runs import RunModel
from auto_ml_flow.client.v1.models.systems import CreateSystemPayload
from auto_ml_flow.handlers.dataset import add_dataset_to
from auto_ml_flow.handlers.experiment import get_or_create_experiment
from auto_ml_flow.handlers.run import run_ended, run_started
from auto_ml_flow.handlers.run_metric import add_metric_to, add_param_to, add_result_to
from auto_ml_flow.handlers.system import create_system
from auto_ml_flow.metrics.monitor import SystemMetricsMonitor
from auto_ml_flow.metrics.system import get_system


class AutoMLFlow:
    _client: AutoMLFlowClient | None = None
    _latest_run: RunModel | None = None
    _experiment: ExperimentModel | None = None
    _monitor: SystemMetricsMonitor | None = None

    @classmethod
    def set_tracking_url(cls, url: str) -> None:
        cls._client = AutoMLFlowClient(base_url=url)

    @classmethod
    def start_experiment(cls, name: str, description: str | None = None) -> None:
        if cls._client is None:
            raise ValueError("Tracking URL is not set. Use 'set_tracking_url' method to set it.")

        cls._experiment = get_or_create_experiment(
            name=name, description=description, client=cls._client
        )

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
        cls._monitor = SystemMetricsMonitor(system=system.id, client=cls._client, interval=0.5)
        cls._monitor.start()

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
            cls._monitor.finish()

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

    @classmethod
    def log_param(cls, key: str, value: str) -> None:
        if cls._client is None:
            raise ValueError("Tracking URL is not set. Use 'set_tracking_url' method to set it.")

        if not cls._latest_run:
            raise ValueError(
                "Not found current run. "
                "First need to call 'with AutoMLFlow.run_manager(experiment)'"
            )

        add_param_to(cls._latest_run, key, value, cls._client)

    @classmethod
    def log_result(cls, key: str, value: float) -> None:
        if cls._client is None:
            raise ValueError("Tracking URL is not set. Use 'set_tracking_url' method to set it.")

        if not cls._latest_run:
            raise ValueError(
                "Not found current run. "
                "First need to call 'with AutoMLFlow.run_manager(experiment)'"
            )

        add_result_to(cls._latest_run, key, value, cls._client)

    @classmethod
    def log_dataset(cls, n_features: int, n_samples: int, file: Any) -> None:
        if cls._client is None:
            raise ValueError("Tracking URL is not set. Use 'set_tracking_url' method to set it.")

        if not cls._latest_run:
            raise ValueError(
                "Not found current run. "
                "First need to call 'with AutoMLFlow.run_manager(experiment)'"
            )

        # Convert various inputs to DataFrame
        if isinstance(file, pd.DataFrame):
            df = file
        elif isinstance(file, np.ndarray):
            df = pd.DataFrame(file)
        elif isinstance(file, (str, bytes)):
            try:
                if isinstance(file, str):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_csv(BytesIO(file))
            except Exception as e:
                raise ValueError(f"Failed to convert file to DataFrame: {e}")
        else:
            raise ValueError("Unsupported file type")

        # Pickle the DataFrame
        with tempfile.NamedTemporaryFile(delete=False, prefix="run_", suffix=".pkl") as temp_file:
            temp_file_name = temp_file.name
            pickle.dump(df, open(temp_file_name, "wb"))
            # Ensure all data is written and set the file position to the beginning
            temp_file.flush()
            temp_file.seek(0)

            try:
                add_dataset_to(cls._latest_run, n_features, n_samples, temp_file, cls._client)
            finally:
                temp_file.close()
