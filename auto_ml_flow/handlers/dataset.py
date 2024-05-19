from typing import IO

from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.client.v1.models.datasets import CreateDatasetPayload, DatasetModel
from auto_ml_flow.client.v1.models.runs import RunModel


def add_dataset_to(
    run: RunModel, n_features: int, n_samples: int, file: IO, client: AutoMLFlowClient
) -> DatasetModel:
    payload = CreateDatasetPayload(n_features=n_features, n_samples=n_samples, run=run.id)

    return client.datasets.create(payload, file=file)
