import os

import psutil

from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.client.v1.models.metrics import DiskMetricModel
from auto_ml_flow.metrics.monitor.base import BaseMetricsMonitor, bytes_to_megabytes


class DiskMonitor(BaseMetricsMonitor):
    def collect_metrics(self) -> None:
        disk_usage = psutil.disk_usage(os.sep)
        self._metrics["disk_usage_percentage"] = disk_usage.percent
        self._metrics["disk_usage_megabytes"] = bytes_to_megabytes(disk_usage.used)
        self._metrics["disk_available_megabytes"] = bytes_to_megabytes(disk_usage.free)

    def log_metrics(self, system: int, client: AutoMLFlowClient) -> None:
        disk_stats = DiskMetricModel(
            usage_megabytes=self.metrics["disk_usage_megabytes"],
            usage_percentage=self.metrics["disk_usage_percentage"],
            available=self.metrics["disk_available_megabytes"],
            system=system,
        )

        client.systems.metrics.disk_stats.create(disk_stats)
