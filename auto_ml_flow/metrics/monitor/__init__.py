import threading
import time

from loguru import logger

from auto_ml_flow.client.v1 import AutoMLFlowClient
from auto_ml_flow.metrics.monitor.cpu import CPUMonitor
from auto_ml_flow.metrics.monitor.disk import DiskMonitor
from auto_ml_flow.metrics.monitor.memory import MemoryMonitor
from auto_ml_flow.metrics.monitor.network import NetworkMonitor


class SystemMetricsMonitor:
    def __init__(self, system: int, client: AutoMLFlowClient, interval: int = 10) -> None:
        """
        Initialize the system metrics monitor.

        Args:
            interval (int): The interval (in seconds) at which to collect metrics.
        """
        self.system = system
        self.client = client
        self.monitors = [CPUMonitor(), DiskMonitor(), NetworkMonitor(), MemoryMonitor()]
        self.interval = interval
        self._shutdown_event = threading.Event()
        self._process: threading.Thread | None = None

    def start(self) -> None:
        """Start the system metrics monitoring in a background thread."""
        if self._process is not None:
            logger.warning("System metrics monitoring is already running.")
            return

        logger.info("Starting system metrics monitoring...")

        self._shutdown_event.clear()
        self._process = threading.Thread(target=self._run)
        self._process.start()

    def _run(self) -> None:
        """Background thread function to collect metrics periodically."""
        while not self._shutdown_event.is_set():
            self.collect_metrics()
            time.sleep(self.interval)

    def collect_metrics(self) -> None:
        """Collect system metrics."""

        for monitor in self.monitors:
            monitor.collect_metrics()
            monitor.log_metrics(system=self.system, client=self.client)

    def finish(self) -> None:
        """Stop monitoring system metrics."""
        if self._process is None:
            logger.warning("System metrics monitoring is not running.")
            return

        self._shutdown_event.set()

        try:
            self._process.join()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error terminating system metrics monitoring process: {e}.")

        self._process = None
