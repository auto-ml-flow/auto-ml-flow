from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.models.run_metrics import (
    CreateRunResultPayload,
    ResultModel,
)


class ResultMetricsClient(BaseClient):
    DEFAULT_PREFIX = "/api/v1/run-results"

    def list(self) -> list[ResultModel]:
        return self._get(f"{self.DEFAULT_PREFIX}/", model=list[ResultModel])

    def retrieve(self, id_: int) -> ResultModel:
        return self._get(f"{self.DEFAULT_PREFIX}/{id_}/", model=ResultModel)

    def create(self, result: CreateRunResultPayload) -> ResultModel:
        return self._post(f"{self.DEFAULT_PREFIX}/", data=result.model_dump(), model=ResultModel)
