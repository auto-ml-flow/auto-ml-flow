import psutil

from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.client.v1.models.metrics import MemoryMetricModel
from auto_ml_flow.metrics.monitor.base import BaseMetricsMonitor, bytes_to_megabytes


class MemoryMonitor(BaseMetricsMonitor):
    def collect_metrics(self) -> None:
        system_memory = psutil.virtual_memory()
        self._metrics["system_memory_usage_megabytes"] = bytes_to_megabytes(system_memory.used)
        self._metrics["system_memory_usage_percentage"] = (
            system_memory.used / system_memory.total * 100
        )

    def log_metrics(self, system: int, client: AutoMLFlowClient) -> None:
        memory_stats = MemoryMetricModel(
            usage_megabytes=self.metrics["system_memory_usage_megabytes"],
            usage_percentage=self.metrics["system_memory_usage_percentage"],
            system=system,
        )

        client.systems.metrics.memory_stats.create(memory_stats)
