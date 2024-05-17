from pydantic import BaseModel


class SystemInfoModel(BaseModel):
    name: str | None = None
    cpu_name: str
    gpu_name: str | None = None
    ram: int
    ram_available: int
    swap: int
    swap_available: int
    load_avg_last_min: float
    load_avg_last_5_min: float
    load_avg_last_15_min: float


class SystemModel(SystemInfoModel):
    id: int


class CreateSystemPayload(SystemInfoModel):
    run: int
