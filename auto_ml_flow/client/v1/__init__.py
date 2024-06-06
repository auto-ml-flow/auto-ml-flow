from requests import Session

from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.api.datasets import DatasetsClient
from auto_ml_flow.client.v1.api.experiments import ExperimentsClient
from auto_ml_flow.client.v1.api.predict import MetaAlgoClient
from auto_ml_flow.client.v1.api.runs import RunsClient
from auto_ml_flow.client.v1.api.systems import SystemsClient


class AutoMLFlowClient(BaseClient):
    def __init__(self, base_url: str | None = None, session: Session | None = None) -> None:
        super().__init__(base_url, session)

        self.experiments = ExperimentsClient(base_url=base_url, session=session)
        self.runs = RunsClient(base_url=base_url, session=session)
        self.systems = SystemsClient(base_url=base_url, session=session)
        self.datasets = DatasetsClient(base_url=base_url, session=session)
        self.meta_algos = MetaAlgoClient(base_url=base_url, session=session)
