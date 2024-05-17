from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.client.v1.models.systems import CreateSystemPayload, SystemModel


def get_system_by(id_: int, client: AutoMLFlowClient) -> SystemModel:
    return client.systems.retrieve(id_)


def create_system(system: CreateSystemPayload, client: AutoMLFlowClient) -> SystemModel:
    return client.systems.create(system)
