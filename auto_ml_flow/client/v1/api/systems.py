from requests import Session

from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.api.metrics import MetricsClient
from auto_ml_flow.client.v1.models.systems import CreateSystemPayload, SystemModel


class SystemsClient(BaseClient):
    def __init__(self, base_url: str | None = None, session: Session | None = None) -> None:
        super().__init__(base_url, session)

        self.metrics = MetricsClient(base_url, session)

    DEFAULT_PREFIX = "/api/v1/systems"

    def list(self) -> list[SystemModel]:
        return self._get(f"{self.DEFAULT_PREFIX}/", model=list[SystemModel])

    def retrieve(self, id_: int) -> SystemModel:
        return self._get(f"{self.DEFAULT_PREFIX}/{id_}/", model=SystemModel)

    def create(self, run: CreateSystemPayload) -> SystemModel:
        return self._post(f"{self.DEFAULT_PREFIX}/", data=run.model_dump(), model=SystemModel)
