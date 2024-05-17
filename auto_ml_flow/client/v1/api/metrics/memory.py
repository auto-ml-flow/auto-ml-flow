from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.models.metrics import MemoryMetricModel


class MemoryMetricsClient(BaseClient):
    DEFAULT_PREFIX = "/api/v1/memory-stats"

    def list(self) -> list[MemoryMetricModel]:
        return self._get(f"{self.DEFAULT_PREFIX}/", model=list[MemoryMetricModel])

    def retrieve(self, id_: int) -> MemoryMetricModel:
        return self._get(f"{self.DEFAULT_PREFIX}/{id_}/", model=MemoryMetricModel)

    def create(self, memory_stat: MemoryMetricModel) -> MemoryMetricModel:
        return self._post(
            f"{self.DEFAULT_PREFIX}/", data=memory_stat.model_dump(), model=MemoryMetricModel
        )
