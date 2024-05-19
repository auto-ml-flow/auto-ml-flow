from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.models.run_metrics import CreateRunParamPayload, ParamModel


class ParamsMetricsClient(BaseClient):
    DEFAULT_PREFIX = "/api/v1/run-params"

    def list(self) -> list[ParamModel]:
        return self._get(f"{self.DEFAULT_PREFIX}/", model=list[ParamModel])

    def retrieve(self, id_: int) -> ParamModel:
        return self._get(f"{self.DEFAULT_PREFIX}/{id_}/", model=ParamModel)

    def create(self, param: CreateRunParamPayload) -> ParamModel:
        return self._post(f"{self.DEFAULT_PREFIX}/", data=param.model_dump(), model=ParamModel)
