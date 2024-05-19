from pydantic import BaseModel


class CreateDatasetPayload(BaseModel):
    n_samples: int
    n_features: int
    run: int


class DatasetModel(CreateDatasetPayload):
    id: int
    file: str
