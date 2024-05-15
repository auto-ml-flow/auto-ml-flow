from contextlib import contextmanager
from datetime import datetime
import traceback
from typing import Any, Generator
from auto_ml_flow.client.v1 import AutoMLFlowClient
from loguru import logger

from auto_ml_flow.client.v1.consts import Status
from auto_ml_flow.client.v1.models.experiments import ExperimentModel
from auto_ml_flow.client.v1.models.runs import CreateRunPayload, PatchRunPayload, RunModel

def get_run_by(id_: int, client: AutoMLFlowClient) -> RunModel:
    return client.runs.retrieve(id_)


def create_run(experiment_id: int, client: AutoMLFlowClient, description: str | None = None) -> RunModel:
    payload = CreateRunPayload(experiment=experiment_id, description=description)
    
    return client.runs.create(payload)


def run_ended(run_id: int, status: Status, duration: float, client: AutoMLFlowClient, traceback: str | None = None) -> RunModel:    
    payload = PatchRunPayload(status=status, duration=duration, traceback=traceback)
    
    logger.debug(f'{run_id=} ended with {status.value=}; {duration=}')
    
    return client.runs.patch(id_=run_id, run=payload)


@contextmanager
def run_manager(client: AutoMLFlowClient, experiment: ExperimentModel, description: str | None = None) -> Generator[RunModel, Any, None]:
    run = create_run(client=client, experiment_id=experiment.id, description=description)
    start_time = datetime.now()
    logger.debug(f'New {run.id=} for experiment {experiment.name} (id={experiment.id}) created')
    
    try:
        yield run
        end_time = datetime.now()
        duration = end_time - start_time
        logger.debug(f'{run.id=} ended with {duration=}')
        run_ended(client=client, run_id=run.id, status=Status.DONE, duration=duration.total_seconds())
    except Exception:
        end_time = datetime.now()
        duration = end_time - start_time
        error_trace = traceback.format_exc()
        logger.debug(error_trace)
        logger.debug(f'{run.id=} ended with {duration=}')
        run_ended(client=client, run_id=run.id, status=Status.FAILED, duration=duration.total_seconds(), traceback=error_trace)
        raise
