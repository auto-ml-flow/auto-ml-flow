from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.models.metrics import DiskMetricModel


class DiskMetricsClient(BaseClient):
    DEFAULT_PREFIX = "/api/v1/disk-stats"

    def list(self) -> list[DiskMetricModel]:
        return self._get(f"{self.DEFAULT_PREFIX}/", model=list[DiskMetricModel])

    def retrieve(self, id_: int) -> DiskMetricModel:
        return self._get(f"{self.DEFAULT_PREFIX}/{id_}/", model=DiskMetricModel)

    def create(self, disk_stat: DiskMetricModel) -> DiskMetricModel:
        return self._post(
            f"{self.DEFAULT_PREFIX}/", data=disk_stat.model_dump(), model=DiskMetricModel
        )
