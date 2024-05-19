import os

import psutil

from auto_ml_flow.client.v1.models.systems import SystemInfoModel
from auto_ml_flow.metrics.cpu import get_cpu_name
from auto_ml_flow.metrics.monitor.base import bytes_to_megabytes


def get_system() -> SystemInfoModel:
    cpu_name = get_cpu_name()
    load_avg_last_min, load_avg_last_5_min, load_avg_last_15_min = os.getloadavg()
    swap = psutil.swap_memory()
    virtual_memory = psutil.virtual_memory()
    ram, ram_available = (
        bytes_to_megabytes(virtual_memory.total, to_int=True),
        bytes_to_megabytes(virtual_memory.available, to_int=True),
    )
    swap_total, swap_available = (
        bytes_to_megabytes(swap.total, to_int=True),
        bytes_to_megabytes(swap.free, to_int=True),
    )

    return SystemInfoModel(
        cpu_name=cpu_name,
        ram=ram,
        ram_available=ram_available,
        swap=swap_total,
        swap_available=swap_available,
        load_avg_last_min=load_avg_last_min,
        load_avg_last_5_min=load_avg_last_5_min,
        load_avg_last_15_min=load_avg_last_15_min,
    )
