from datetime import datetime

from pydantic import BaseModel


class ExperimentModel(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    name: str
    description: str | None = None


class CreateExperimentPayload(BaseModel):
    name: str
    description: str | None = None
