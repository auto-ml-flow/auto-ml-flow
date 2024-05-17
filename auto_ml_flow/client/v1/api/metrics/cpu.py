from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.models.metrics import CPUMetricModel


class CPUMetricsClient(BaseClient):
    DEFAULT_PREFIX = "/api/v1/cpu-stats"

    def list(self) -> list[CPUMetricModel]:
        return self._get(f"{self.DEFAULT_PREFIX}/", model=list[CPUMetricModel])

    def retrieve(self, id_: int) -> CPUMetricModel:
        return self._get(f"{self.DEFAULT_PREFIX}/{id_}/", model=CPUMetricModel)

    def create(self, cpu_stat: CPUMetricModel) -> CPUMetricModel:
        return self._post(
            f"{self.DEFAULT_PREFIX}/", data=cpu_stat.model_dump(), model=CPUMetricModel
        )
