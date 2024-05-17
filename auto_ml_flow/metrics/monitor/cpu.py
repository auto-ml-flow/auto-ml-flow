import psutil

from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.client.v1.models.metrics import CPUMetricModel
from auto_ml_flow.metrics.monitor.base import BaseMetricsMonitor


class CPUMonitor(BaseMetricsMonitor):
    def collect_metrics(self) -> None:
        self._metrics["cpu_utilization_percentage"] = psutil.cpu_percent()

    def log_metrics(self, system: int, client: AutoMLFlowClient) -> None:
        cpu_stats = CPUMetricModel(
            utilization=self.metrics["cpu_utilization_percentage"], system=system
        )

        client.systems.metrics.cpu_stats.create(cpu_stats)
