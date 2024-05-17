"""Base class of system metrics monitor."""

import abc

from auto_ml_flow.client.v1 import AutoMLFlowClient


class BaseMetricsMonitor(abc.ABC):
    """Base class of system metrics monitor."""

    def __init__(self) -> None:
        self._metrics: dict[str, float] = {}

    @abc.abstractmethod
    def collect_metrics(self) -> None:
        """Method to collect metrics.

        Subclass should implement this method to collect metrics and store in `self._metrics`.
        """
        ...

    @abc.abstractmethod
    def log_metrics(self, system: int, client: AutoMLFlowClient) -> None:
        """Method to send metrics to server

        Subclass should implement this method to collect metrics, and call client method
        """
        ...

    @property
    def metrics(self) -> dict[str, float]:
        return self._metrics

    def clear_metrics(self) -> None:
        self._metrics.clear()
