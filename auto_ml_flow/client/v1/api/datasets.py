from typing import IO

from requests import Session

from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.models.datasets import CreateDatasetPayload, DatasetModel


class DatasetsClient(BaseClient):
    def __init__(self, base_url: str | None = None, session: Session | None = None) -> None:
        super().__init__(base_url, session)

    DEFAULT_PREFIX = "/api/v1/datasets"

    def list(self) -> list[DatasetModel]:
        return self._get(f"{self.DEFAULT_PREFIX}/", model=list[DatasetModel])

    def retrieve(self, id_: int) -> DatasetModel:
        return self._get(f"{self.DEFAULT_PREFIX}/{id_}/", model=DatasetModel)

    def create(self, dataset: CreateDatasetPayload, file: IO) -> DatasetModel:
        return self._post(
            f"{self.DEFAULT_PREFIX}/",
            data=dataset.model_dump(),
            files={"file": (file.name, file)},
            model=DatasetModel,
        )
