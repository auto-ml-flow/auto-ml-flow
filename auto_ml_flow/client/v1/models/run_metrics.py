from datetime import datetime
from pydantic import BaseModel


class RunMetric(BaseModel):
    key: str
    value: float
    created_at: datetime
    updated_at: datetime
    run: int

class CreateRunMetricPayload(BaseModel):
    key: str
    value: float
    run: int
