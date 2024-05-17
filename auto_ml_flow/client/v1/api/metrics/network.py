from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.models.metrics import NetworkMetricModel


class NetworkMetricsClient(BaseClient):
    DEFAULT_PREFIX = "/api/v1/network-stats"

    def list(self) -> list[NetworkMetricModel]:
        return self._get(f"{self.DEFAULT_PREFIX}/", model=list[NetworkMetricModel])

    def retrieve(self, id_: int) -> NetworkMetricModel:
        return self._get(f"{self.DEFAULT_PREFIX}/{id_}/", model=NetworkMetricModel)

    def create(self, network_stat: NetworkMetricModel) -> NetworkMetricModel:
        return self._post(
            f"{self.DEFAULT_PREFIX}/", data=network_stat.model_dump(), model=NetworkMetricModel
        )
