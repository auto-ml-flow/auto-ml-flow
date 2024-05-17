from requests import Session

from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.api.metrics.cpu import CPUMetricsClient
from auto_ml_flow.client.v1.api.metrics.disk import DiskMetricsClient
from auto_ml_flow.client.v1.api.metrics.memory import MemoryMetricsClient
from auto_ml_flow.client.v1.api.metrics.network import NetworkMetricsClient


class MetricsClient(BaseClient):
    def __init__(self, base_url: str | None = None, session: Session | None = None) -> None:
        super().__init__(base_url, session)

        self.cpu_stats = CPUMetricsClient(base_url, session)
        self.network_stats = NetworkMetricsClient(base_url, session)
        self.memory_stats = MemoryMetricsClient(base_url, session)
        self.disk_stats = DiskMetricsClient(base_url, session)
