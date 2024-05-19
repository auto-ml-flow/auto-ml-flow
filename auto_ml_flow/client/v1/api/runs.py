from requests import Session

from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.api.param_metrics import ParamsMetricsClient
from auto_ml_flow.client.v1.api.result_metrics import ResultMetricsClient
from auto_ml_flow.client.v1.api.run_metrics import RunMetricsClient
from auto_ml_flow.client.v1.models.runs import (
    CreateRunPayload,
    PatchRunPayload,
    RunModel,
)


class RunsClient(BaseClient):
    def __init__(self, base_url: str | None = None, session: Session | None = None) -> None:
        super().__init__(base_url, session)

        self.metrics = RunMetricsClient(base_url, session=session)
        self.params = ParamsMetricsClient(base_url, session=session)
        self.results = ResultMetricsClient(base_url, session=session)

    DEFAULT_PREFIX = "/api/v1/runs"

    def list(self) -> list[RunModel]:
        return self._get(f"{self.DEFAULT_PREFIX}/", model=list[RunModel])

    def retrieve(self, id_: int) -> RunModel:
        return self._get(f"{self.DEFAULT_PREFIX}/{id_}/", model=RunModel)

    def create(self, run: CreateRunPayload) -> RunModel:
        return self._post(f"{self.DEFAULT_PREFIX}/", data=run.model_dump(), model=RunModel)

    def patch(self, id_: int, run: PatchRunPayload) -> RunModel:
        return self._patch(f"{self.DEFAULT_PREFIX}/{id_}/", data=run.model_dump(), model=RunModel)
