"""Metrics configuration and utilities for model evaluation and tuning."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import torchmetrics

from autoemulate.core.types import MetricLike


class Metric:
    """Configuration for a single metric.

    Parameters
    ----------
    metric : MetricLike
        The torchmetrics metric class or partial.
    name : str
        Display name for the metric.
    maximize : bool
        Whether higher values are better. Defaults to True.
    """

    metric: MetricLike
    name: str
    maximize: bool

    def __repr__(self) -> str:
        """Return the string representation of the MetricConfig."""
        return f"MetricConfig(name={self.name}, maximize={self.maximize})"


class TorchMetrics(Metric):
    """Configuration for a single torchmetrics metric.

    Parameters
    ----------
    metric : MetricLike
        The torchmetrics metric class or partial.
    name : str
        Display name for the metric. If None, uses the class name of the metric.
    maximize : bool
        Whether higher values are better.
    """

    def __init__(
        self,
        metric: MetricLike,
        name: str,
        maximize: bool,
    ):
        self.metric = metric
        self.name = name
        self.maximize = maximize


R2 = TorchMetrics(
    metric=torchmetrics.R2Score,
    name="r2",
    maximize=True,
)

RMSE = TorchMetrics(
    metric=partial(torchmetrics.MeanSquaredError, squared=False),
    name="rmse",
    maximize=False,
)

MSE = TorchMetrics(
    metric=torchmetrics.MeanSquaredError,
    name="mse",
    maximize=False,
)

MAE = TorchMetrics(
    metric=torchmetrics.MeanAbsoluteError,
    name="mae",
    maximize=False,
)

AVAILABLE_METRICS = {
    "r2": R2,
    "rmse": RMSE,
    "mse": MSE,
    "mae": MAE,
}


def get_metric_config(
    metric: str | TorchMetrics,
) -> TorchMetrics:
    """Convert various metric specifications to MetricConfig.

    Parameters
    ----------
    metric : str | type[torchmetrics.Metric] | partial[torchmetrics.Metric] | Metric
        The metric specification. Can be:
        - A string shortcut like "r2", "rmse", "mse", "mae"
        - A Metric instance (returned as-is)

    Returns
    -------
    TorchMetrics
        The metric configuration.

    Raises
    ------
    ValueError
        If the metric specification is invalid or name is not provided when required.


    """
    # If already a TorchMetric, return as-is
    if isinstance(metric, TorchMetrics):
        return metric

    if isinstance(metric, str):
        if metric.lower() in AVAILABLE_METRICS:
            return AVAILABLE_METRICS[metric.lower()]
        raise ValueError(
            f"Unknown metric shortcut '{metric}'. "
            f"Available options: {list(AVAILABLE_METRICS.keys())}"
        )
    # Handle unsupported types
    raise ValueError(
        f"Unsupported metric type: {type(metric).__name__}. "
        "Metric must be a string shortcut or a MetricConfig instance."
    )


def get_metric_configs(
    metrics: Sequence[str | TorchMetrics],
) -> list[TorchMetrics]:
    """Convert a list of metric specifications to MetricConfig objects.

    Parameters
    ----------
    metrics : Sequence[str | TorchMetrics]
        Sequence of metric specifications.

    Returns
    -------
    list[TorchMetrics]
        List of metric configurations.
    """
    result_metrics = []

    for m in metrics:
        config = get_metric_config(m) if isinstance(m, (str | TorchMetrics)) else m
        result_metrics.append(config)

    return result_metrics
