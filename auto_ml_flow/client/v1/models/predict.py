from pydantic import BaseModel


class MetaAlgoFeatures(BaseModel):
    system_ram: int
    system_swap: int
    system_swap_available: int
    system_load_avg_last_min: float
    system_load_avg_last_5_min: float
    system_load_avg_last_15_min: float
    dataset_n_features: int
    dataset_n_samples: int
    avg_memory_usage_megabytes: float
    avg_memory_usage_percentage: float
    avg_cpu_utilization: float
    avg_disk_usage_percentage: float
    avg_disk_usage_megabytes: float
    avg_disk_available: float
    sum_network_receive_megabytes: float
    sum_network_transmit_megabytes: float


class MetaAlgoPredictions(BaseModel):
    duration: float
