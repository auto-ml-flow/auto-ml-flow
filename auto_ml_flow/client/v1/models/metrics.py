from pydantic import BaseModel


class MemoryMetricModel(BaseModel):
    usage_megabytes: float
    usage_percentage: float
    system: int


class CPUMetricModel(BaseModel):
    utilization: float
    system: int


class DiskMetricModel(BaseModel):
    usage_percentage: float
    usage_megabytes: float
    available: float
    system: int


class NetworkMetricModel(BaseModel):
    receive_megabytes: float
    transmit_megabytes: float
    system: int
