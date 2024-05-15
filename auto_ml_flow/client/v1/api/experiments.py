from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.models.experiments import CreateExperimentPayload, ExperimentModel

class ExperimentsClient(BaseClient):
    DEFAULT_PREFIX = '/api/v1/experiments'
    
    def list(self) -> list[ExperimentModel]:
        return self._get(f'{self.DEFAULT_PREFIX}/', model=list[ExperimentModel])
    
    def retrieve(self, name: str) -> ExperimentModel:
        return self._get(f'{self.DEFAULT_PREFIX}/{name}/', model=ExperimentModel)

    def create(self, experiment: CreateExperimentPayload) -> ExperimentModel:
        return self._post(f'{self.DEFAULT_PREFIX}/', data=experiment.model_dump(), model=ExperimentModel)
