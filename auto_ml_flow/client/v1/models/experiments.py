from datetime import datetime
from pydantic import BaseModel

class ExperimentModel(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    name: str

class CreateExperimentPayload(BaseModel):
    name: str
