from cpuinfo import get_cpu_info


def get_cpu_name() -> str:
    cpu_info = get_cpu_info()

    return cpu_info["brand_raw"]
