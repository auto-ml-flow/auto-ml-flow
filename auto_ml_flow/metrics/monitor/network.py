import psutil

from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.client.v1.models.metrics import NetworkMetricModel
from auto_ml_flow.metrics.monitor.base import BaseMetricsMonitor


class NetworkMonitor(BaseMetricsMonitor):
    def __init__(self) -> None:
        super().__init__()
        self._set_initial_metrics()

    def _set_initial_metrics(self) -> None:
        # Set initial network usage metrics. `psutil.net_io_counters()` counts the stats since the
        # system boot, so to set network usage metrics as 0 when we start logging, we need to keep
        # the initial network usage metrics.
        network_usage = psutil.net_io_counters()
        self._initial_receive_megabytes = network_usage.bytes_recv / 1e6
        self._initial_transmit_megabytes = network_usage.bytes_sent / 1e6

    def collect_metrics(self) -> None:
        network_usage = psutil.net_io_counters()
        self._metrics["network_receive_megabytes"] = (
            network_usage.bytes_recv / 1e6 - self._initial_receive_megabytes
        )
        self._metrics["network_transmit_megabytes"] = (
            network_usage.bytes_sent / 1e6 - self._initial_transmit_megabytes
        )

    def log_metrics(self, system: int, client: AutoMLFlowClient) -> None:
        network_stats = NetworkMetricModel(
            receive_megabytes=self.metrics["network_receive_megabytes"],
            transmit_megabytes=self.metrics["network_transmit_megabytes"],
            system=system,
        )

        client.systems.metrics.network_stats.create(network_stats)
