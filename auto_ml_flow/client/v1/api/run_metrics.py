from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.models.run_metrics import CreateRunMetricPayload, RunMetric

class RunMetricsClient(BaseClient):
    DEFAULT_PREFIX = '/api/v1/run-metrics'
    
    def list(self) -> list[RunMetric]:
        return self._get(f'{self.DEFAULT_PREFIX}/', model=list[RunMetric])
    
    def retrieve(self, id_: int) -> RunMetric:
        return self._get(f'{self.DEFAULT_PREFIX}/{id_}/', model=RunMetric)

    def create(self, run: CreateRunMetricPayload) -> RunMetric:
        return self._post(f'{self.DEFAULT_PREFIX}/', data=run.model_dump(), model=RunMetric)
