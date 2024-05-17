from auto_ml_flow.client.exceptions import ClientNotFoundError
from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.client.v1.models.experiments import (
    CreateExperimentPayload,
    ExperimentModel,
)


def get_experiment_by(name: str, client: AutoMLFlowClient) -> ExperimentModel:
    return client.experiments.retrieve(name)


def create_experiment(name: str, client: AutoMLFlowClient) -> ExperimentModel:
    payload = CreateExperimentPayload(name=name)

    return client.experiments.create(payload)


def get_or_create_experiment(name: str, client: AutoMLFlowClient) -> ExperimentModel:
    try:
        return get_experiment_by(name, client)
    except ClientNotFoundError:
        return create_experiment(name, client)
