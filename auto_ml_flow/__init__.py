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

from auto_ml_flow.client.exceptions import ClientServerError
from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.client.v1.consts import Status
from auto_ml_flow.client.v1.models.experiments import ExperimentModel
from auto_ml_flow.client.v1.models.predict import MetaAlgoFeatures
from auto_ml_flow.client.v1.models.runs import RunModel
from auto_ml_flow.client.v1.models.systems import CreateSystemPayload, SystemInfoModel
from auto_ml_flow.handlers.dataset import add_dataset_to
from auto_ml_flow.handlers.experiment import get_or_create_experiment
from auto_ml_flow.handlers.run import run_ended, run_started
from auto_ml_flow.handlers.run_metric import add_metric_to, add_param_to, add_result_to
from auto_ml_flow.handlers.system import create_system
from auto_ml_flow.metrics.monitor import SystemMetricsMonitor
from auto_ml_flow.metrics.monitor.cpu import CPUMonitor
from auto_ml_flow.metrics.monitor.disk import DiskMonitor
from auto_ml_flow.metrics.monitor.memory import MemoryMonitor
from auto_ml_flow.metrics.monitor.network import NetworkMonitor
from auto_ml_flow.metrics.system import get_system


class AutoMLFlow:
    _client: AutoMLFlowClient | None = None
    _latest_run: RunModel | None = None
    _experiment: ExperimentModel | None = None
    _monitor: SystemMetricsMonitor | None = None
    _n_features: int = 0
    _n_samples: int = 0
    _system_info: SystemInfoModel | None = None
    _predicted_time: float | None = None

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

        system_info = cls._system_info = (
            get_system()
        )  # before run creation because it pretty hard task
        run = run_started(
            client=cls._client, experiment_id=cls._experiment.id, description=description
        )

        start_time = datetime.now()
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
            run_ended(
                client=cls._client,
                run_id=run.id,
                status=Status.DONE,
                duration=duration.total_seconds(),
                predicted_time=cls._predicted_time,
            )
        except Exception:
            end_time = datetime.now()

            duration = end_time - start_time

            error_trace = traceback.format_exc()
            run_ended(
                client=cls._client,
                run_id=run.id,
                status=Status.FAILED,
                duration=duration.total_seconds(),
                traceback=error_trace,
                predicted_time=cls._predicted_time,
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
    def predict_training_time(cls) -> None:
        if cls._client is None:
            raise ValueError("Tracking URL is not set. Use 'set_tracking_url' method to set it.")

        if not cls._latest_run:
            raise ValueError(
                "Not found current run. "
                "First need to call 'with AutoMLFlow.run_manager(experiment)'"
            )

        if not cls._n_samples or not cls._n_features:
            raise ValueError(
                "Before predict training time "
                "Need to log dataset 'with AutoMLFlow.log_dataset(n_features, n_samples, X)'"
            )

        if not cls._system_info:
            raise ValueError("Some error happens. Failed to empty system info!")
        # TODO: add handler

        metrics = {}

        monitors = [CPUMonitor(), DiskMonitor(), NetworkMonitor(), MemoryMonitor()]
        for monitor in monitors:
            monitor.collect_metrics()
            metrics.update(monitor.metrics)

        metrics.update(cls._system_info.model_dump())
        features = MetaAlgoFeatures(
            system_ram=metrics["ram"],
            system_swap=metrics["swap"],
            system_swap_available=metrics["swap_available"],
            system_load_avg_last_min=metrics["load_avg_last_min"],
            system_load_avg_last_5_min=metrics["load_avg_last_5_min"],
            system_load_avg_last_15_min=metrics["load_avg_last_15_min"],
            avg_memory_usage_megabytes=metrics["system_memory_usage_megabytes"],
            avg_memory_usage_percentage=metrics["system_memory_usage_percentage"],
            avg_cpu_utilization=metrics["cpu_utilization_percentage"],
            avg_disk_usage_percentage=metrics["disk_usage_percentage"],
            avg_disk_usage_megabytes=metrics["disk_usage_megabytes"],
            avg_disk_available=metrics["disk_available_megabytes"],
            sum_network_receive_megabytes=metrics["network_receive_megabytes"],
            sum_network_transmit_megabytes=metrics["network_transmit_megabytes"],
            dataset_n_samples=cls._n_samples,
            dataset_n_features=cls._n_features,
        )
        try:
            cls._predicted_time = cls._client.meta_algos.predict(features)
        except ClientServerError:
            logger.warning("To less data for predict training time!")
        else:
            logger.info(f"The current launch will be pre-completed after: {cls._predicted_time}")

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

        cls._n_features = n_features
        cls._n_samples = n_samples
