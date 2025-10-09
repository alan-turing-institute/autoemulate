"""Metrics configuration and utilities for model evaluation and tuning."""

from __future__ import annotations

from functools import partial
from typing import Any

import torchmetrics


class MetricConfig:
    """Configuration for a single metric.

    Parameters
    ----------
    metric : type[torchmetrics.Metric] | partial[torchmetrics.Metric]
        The torchmetrics metric class or partial.
    name : str
        Display name for the metric.
    maximize : bool
        Whether higher values are better. Defaults to True.
    """

    def __init__(
        self,
        metric: type[torchmetrics.Metric] | partial[torchmetrics.Metric],
        name: str,
        maximize: bool = True,
    ):
        self.metric = metric
        self.name = name
        self.maximize = maximize

    def __repr__(self) -> str:
        return f"MetricConfig(name={self.name}, maximize={self.maximize})"


R2 = MetricConfig(
    metric=torchmetrics.R2Score,
    name="r2",
    maximize=True,
)

RMSE = MetricConfig(
    metric=partial(torchmetrics.MeanSquaredError, squared=False),
    name="rmse",
    maximize=False,
)

MSE = MetricConfig(
    metric=torchmetrics.MeanSquaredError,
    name="mse",
    maximize=False,
)

MAE = MetricConfig(
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
    metric: str | MetricConfig,
) -> MetricConfig:
    """Convert various metric specifications to MetricConfig.

    Parameters
    ----------
    metric : str | type[torchmetrics.Metric] | partial[torchmetrics.Metric] | MetricConfig
        The metric specification. Can be:
        - A string shortcut like "r2", "rmse", "mse", "mae"
        - A MetricConfig instance (returned as-is)

    Returns
    -------
    MetricConfig
        The metric configuration.

    Raises
    ------
    ValueError
        If the metric specification is invalid or name is not provided when required.


    """
    # If already a MetricConfig, return as-is
    if isinstance(metric, MetricConfig):
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
    metrics: (list[str | MetricConfig]),
) -> list[MetricConfig]:
    """Convert a list of metric specifications to MetricConfig objects.

    Parameters
    ----------
    metrics : list | None
        List of metric specifications. If None, returns default evaluation metrics.

    Returns
    -------
    list[MetricConfig]
        List of metric configurations.
    """
    result_metrics = []

    for m in metrics:
        config = get_metric_config(m) if isinstance(m, (str, MetricConfig)) else m
        result_metrics.append(config)

    return result_metrics
