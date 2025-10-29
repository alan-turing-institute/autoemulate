"""Metrics configuration and utilities for model evaluation and tuning."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from functools import partial

import torchmetrics
from einops import rearrange
from torchmetrics.regression.crps import ContinuousRankedProbabilityScore

from autoemulate.core.types import (
    DistributionLike,
    OutputLike,
    TensorLike,
    TorchMetricsLike,
)


class Metric:
    """Configuration for a single metric.

    Parameters
    ----------
    name : str
        Display name for the metric.
    maximize : bool
        Whether higher values are better. Defaults to True.
    """

    name: str
    maximize: bool

    def __repr__(self) -> str:
        """Return the string representation of the MetricConfig."""
        return f"MetricConfig(name={self.name}, maximize={self.maximize})"

    @abstractmethod
    def __call__(self, y_pred: OutputLike, y_true: TensorLike) -> TensorLike:
        """Calculate metric."""


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
        metric: TorchMetricsLike,
        name: str,
        maximize: bool,
    ):
        self.metric = metric
        self.name = name
        self.maximize = maximize

    def __call__(self, y_pred: OutputLike, y_true: TensorLike) -> TensorLike:
        """Calculate metric."""
        if not isinstance(y_pred, TensorLike):
            raise ValueError(f"Metric not implemented for y_pred ({type(y_pred)})")
        if not isinstance(y_true, TensorLike):
            raise ValueError(f"Metric not implemented for y_true ({type(y_true)})")

        metric = self.metric()
        metric.to(y_pred.device)
        # Assume first dim is a batch dim, flatten others for metric calculation
        metric.update(y_pred.flatten(start_dim=1), y_true.flatten(start_dim=1))
        return metric.compute()


class ProbabilisticMetric(Metric):
    """Base class for probabilistic metrics."""

    @abstractmethod
    def __call__(self, y_pred: OutputLike, y_true: TensorLike) -> TensorLike:
        """Calculate metric."""


class CRPS(ProbabilisticMetric):
    """Continuous Ranked Probability Score (CRPS) metric.

    Parameters
    ----------
    name : str
        Display name for the metric.
    maximize : bool
        Whether higher values are better. Defaults to False.
    """

    name: str = "crps"
    maximize: bool = False

    def __call__(
        self, y_pred: OutputLike, y_true: TensorLike, n_samples: int = 1000
    ) -> TensorLike:
        """Calculate CRPS metric.

        The metric can handle both deterministic predictions (tensors) and probabilistic
        predictions.

        Aggregation across batch and target dimensions is performed by flattening such
        that the sum of scores is taken across all samples for each point.

        Parameters
        ----------
        y_pred: OutputLike
            Predicted outputs. Can be a tensor or a distribution. If `y_pred` is a
            tensor of shape (batch_size, *(target_shape)), it is treated as
            a deterministic prediction and reduces the metric calculation to mean
            absolute error.
            If `y_pred` is a tensor of shape
            `(batch_size, *(target_shape),  n_samples)`, it is treated as a
            probabilistic prediction and the metric is computed across the samples.
            If `y_pred` is a distribution, then `n_samples` are drawn from the predicted
            distribution to estimate the CRPS.
        y_true: TensorLike
            True target values.
        n_samples: int
            Number of samples to draw from the predicted distribution if `y_pred` is a
            distribution. Defaults to 1000.

        """
        if not isinstance(y_true, TensorLike):
            raise ValueError(f"Metric not implemented for y_true ({type(y_true)})")

        crps_metric = ContinuousRankedProbabilityScore()
        crps_metric.to(y_true.device)

        # Deterministic predictions case
        if (isinstance(y_pred, TensorLike) and y_pred.dim() == y_true.dim()) or (
            isinstance(y_pred, TensorLike) and y_pred.dim() == y_true.dim() + 1
        ):
            samples = y_pred
        # Distribution case
        elif isinstance(y_pred, DistributionLike):
            # Move sample dim to end
            samples = rearrange(y_pred.sample((n_samples,)), "s b ... -> b ... s")
            print(samples.shape, y_true.shape)
            assert samples.shape[:-1] == y_true.shape, (
                f"predictive distribution samples shape {samples.shape} does not match "
                f"y_true shape {y_true.shape}  "
            )
        # Otherwise, raise error
        else:
            if isinstance(y_pred, TensorLike) and isinstance(y_true, TensorLike):
                msg = (
                    f"Metric not implemented for y_pred shape ({y_pred.shape}) given "
                    f"y_true shape ({y_true.shape})"
                )
                raise ValueError(msg)
            msg = (
                f"Metric not implemented for y_pred ({type(y_pred)}) and y_true "
                f"({type(y_true)})"
            )
            raise ValueError(msg)

        # Reshape samples and y_true to (-1, n_samples) and (-1,) respectively, compute
        samples = samples.flatten(start_dim=0, end_dim=-2)
        return crps_metric(samples, y_true.flatten())


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
