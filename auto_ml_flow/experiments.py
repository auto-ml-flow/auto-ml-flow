from contextlib import contextmanager
from typing import Any, Generator
import uuid
from auto_ml_flow.client.v1.models.experiments import CreateExperimentPayload, ExperimentModel
from auto_ml_flow.client.v1 import AutoMLFlowClient
from loguru import logger

def get_experiment_by(name: str, client: AutoMLFlowClient) -> ExperimentModel:
    return client.experiments.retrieve(name)

def create_experiment(name: str, client: AutoMLFlowClient) -> ExperimentModel:
    payload = CreateExperimentPayload(name=name)
    
    return client.experiments.create(payload)

@contextmanager
def experiment_manager(client: AutoMLFlowClient, experiment_name: str | None = None) -> Generator[ExperimentModel, Any, None]:
    try:
        if not experiment_name:
            raise Exception # why so ugly, max?
        
        yield get_experiment_by(experiment_name, client)
    except:
        if experiment_name is None:
            experiment_name = str(uuid.uuid4())
            logger.warning(f'The name of the experiment is critical. Please specify it, it will be easier for you to work with. Right now the generated name is used {experiment_name}')
                
        experiment = create_experiment(experiment_name, client)

        yield experiment
