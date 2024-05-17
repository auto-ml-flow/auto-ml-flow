from loguru import logger

from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.client.v1.consts import Status
from auto_ml_flow.client.v1.models.runs import (
    CreateRunPayload,
    PatchRunPayload,
    RunModel,
)


def get_run_by(id_: int, client: AutoMLFlowClient) -> RunModel:
    return client.runs.retrieve(id_)


def run_started(
    experiment_id: int, client: AutoMLFlowClient, description: str | None = None
) -> RunModel:
    payload = CreateRunPayload(experiment=experiment_id, description=description)

    return client.runs.create(payload)


def run_ended(
    run_id: int,
    status: Status,
    duration: float,
    client: AutoMLFlowClient,
    traceback: str | None = None,
) -> RunModel:
    payload = PatchRunPayload(status=status, duration=duration, traceback=traceback)

    logger.debug(f"{run_id=} ended with {status.value=}; {duration=}")

    return client.runs.patch(id_=run_id, run=payload)
