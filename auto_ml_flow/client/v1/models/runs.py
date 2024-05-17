from datetime import datetime

from pydantic import BaseModel

from auto_ml_flow.client.v1.consts import Status


class RunModel(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    duration: float | None
    experiment: int
    traceback: str | None
    description: str | None = None


class CreateRunPayload(BaseModel):
    experiment: int
    description: str | None = None


class PatchRunPayload(BaseModel):
    status: Status
    duration: float | None = None
    traceback: str | None = None
    description: str | None = None
