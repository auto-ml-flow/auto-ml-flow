from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.client.v1.models.run_metrics import (
    CreateRunMetricPayload,
    CreateRunParamPayload,
    CreateRunResultPayload,
    RunMetric,
)
from auto_ml_flow.client.v1.models.runs import RunModel


def get_run_metrics_by(id_: int, client: AutoMLFlowClient) -> RunMetric:
    return client.runs.metrics.retrieve(id_)


def add_metric_to(run: RunModel, key: str, value: float, client: AutoMLFlowClient) -> RunMetric:
    payload = CreateRunMetricPayload(key=key, value=value, run=run.id)

    return client.runs.metrics.create(payload)


def add_result_to(run: RunModel, key: str, value: float, client: AutoMLFlowClient) -> RunMetric:
    payload = CreateRunResultPayload(key=key, value=value, run=run.id)

    return client.runs.results.create(payload)


def add_param_to(run: RunModel, key: str, value: str, client: AutoMLFlowClient) -> RunMetric:
    payload = CreateRunParamPayload(key=key, value=value, run=run.id)

    return client.runs.params.create(payload)
