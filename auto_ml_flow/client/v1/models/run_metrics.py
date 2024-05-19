from datetime import datetime
from typing import Any

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


class ParamModel(RunMetric):
    value: Any


class CreateRunParamPayload(BaseModel):
    key: str
    value: str | float
    run: int


class ResultModel(RunMetric): ...


class CreateRunResultPayload(CreateRunMetricPayload): ...
