"""Metrics configuration and utilities for model evaluation and tuning."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial, total_ordering
from typing import Literal

import torch
import torchmetrics
from einops import rearrange
from torch.distributions import Independent
from torchmetrics.regression.crps import ContinuousRankedProbabilityScore

from autoemulate.core.types import (
    DistributionLike,
    OutputLike,
    TensorLike,
    TorchMetricsLike,
)


@dataclass
class MetricParams:
    """
    Parameters for metric calculations.

    Attributes
    ----------
    n_samples: int
        Number of samples to draw from the predicted distribution if `y_pred` is a
        distribution. Defaults to 1000.
    y_train: TensorLike | None
        Training target values. In MSLL used to parameterize the trivial model for
        standardization. If None, mean log loss is computed without standardization.
        Defaults to None.
    reduction: Literal["mean", "none"]
        Reduction method to apply to the final metric scores computer per task.
        Options are 'mean' or 'none'. Defaults to 'mean'.
    """

    n_samples: int = 1000
    y_train: TensorLike | None = None
    reduction: Literal["mean", "none"] = "mean"
    metric_kwargs: dict | None = None  # supports subclasses with arbitrary new kwargs


@total_ordering
class Metric:
    """Configuration for a single metric.

    Parameters
    ----------
    name: str
        Display name for the metric.
    maximize: bool
        Whether higher values are better. Defaults to True.
    """

    name: str
    maximize: bool

    def __repr__(self) -> str:
        """Representation of the Metric."""
        return f"Metric(name={self.name}, maximize={self.maximize})"

    def __str__(self):
        """Metric when formatted as a string."""
        return self.name

    def __eq__(self, other: object) -> bool:
        """Check equality based on metric name."""
        if not isinstance(other, Metric):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        """Return hash based on metric name."""
        return hash(self.name)

    def __lt__(self, other: Metric) -> bool:
        """Compare metrics based on their str name."""
        return self.name < other.name

    @abstractmethod
    def __call__(
        self,
        y_pred: OutputLike,
        y_true: TensorLike,
        metric_params: MetricParams | None = None,
    ) -> TensorLike:
        """Calculate metric."""


class TorchMetrics(Metric):
    """Configuration for a single torchmetrics metric.

    Parameters
    ----------
    metric: MetricLike
        The torchmetrics metric class or partial.
    name: str
        Display name for the metric. If None, uses the class name of the metric.
    maximize: bool
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

    def __call__(
        self,
        y_pred: OutputLike,
        y_true: TensorLike,
        metric_params: MetricParams | None = None,
    ) -> TensorLike:
        """Calculate metric."""
        if not isinstance(y_pred, OutputLike):
            raise ValueError(f"Metric not implemented for y_pred ({type(y_pred)})")
        if not isinstance(y_true, TensorLike):
            raise ValueError(f"Metric not implemented for y_true ({type(y_true)})")

        if metric_params is None:
            metric_params = MetricParams()

        # Handle probabilistic predictions
        if isinstance(y_pred, DistributionLike):
            try:
                y_pred = y_pred.mean
            except Exception:
                y_pred = y_pred.rsample(torch.Size([metric_params.n_samples])).mean(
                    dim=0
                )
        metric = self.metric()
        metric.to(y_pred.device)

        # Assume first dim is a batch dim if >=2D, flatten others for metric calculation
        metric.update(
            y_pred.flatten(start_dim=1) if y_pred.ndim > 1 else y_pred,
            y_true.flatten(start_dim=1) if y_true.ndim > 1 else y_true,
        )
        return metric.compute()


class ProbabilisticMetric(Metric):
    """Base class for probabilistic metrics."""

    @abstractmethod
    def __call__(
        self,
        y_pred: OutputLike,
        y_true: TensorLike,
        metric_params: MetricParams | None = None,
    ) -> TensorLike:
        """Calculate metric."""


class CRPSMetric(ProbabilisticMetric):
    """Continuous Ranked Probability Score (CRPS) metric.

    CRPS is a scoring rule for evaluating probabilistic predictions. It reduces to mean
    absolute error (MAE) for deterministic predictions and generalizes to distributions
    by measuring the integral difference between predicted and actual CDFs.

    The metric aggregates over batch and target dimensions by computing the mean
    CRPS across all scalar outputs, making it comparable across different batch
    sizes and output dimensions.

    Attributes
    ----------
    name: str
        Display name for the metric.
    maximize: bool
        Whether higher values are better. False for CRPS (lower is better).
    """

    name: str = "crps"
    maximize: bool = False

    def __call__(
        self,
        y_pred: OutputLike,
        y_true: TensorLike,
        metric_params: MetricParams | None = None,
    ) -> TensorLike:
        """Calculate CRPS metric.

        The metric handles both deterministic predictions (tensors) and probabilistic
        predictions (tensors of samples or distributions).

        Aggregation across batch and target dimensions is performed by computing the
        mean CRPS across all scalar outputs. This makes the metric comparable across
        different batch sizes and target dimensions.

        Parameters
        ----------
        y_pred: OutputLike
            Predicted outputs. Can be a tensor or a distribution.
            - If tensor with shape `(batch_size, *target_shape)`: treated as
            deterministic prediction (reduces to MAE).
            - If tensor with shape `(batch_size, *target_shape, n_samples)`: treated as
            samples from a probabilistic prediction.
            - If distribution: `n_samples` are drawn to estimate CRPS.
        y_true: TensorLike
            True target values of shape `(batch_size, *target_shape)`.
        metric_params: MetricParams
            Metric parameters including: n_samples.

        Returns
        -------
        TensorLike
            Mean CRPS score across all batch elements and target dimensions.

        Raises
        ------
        ValueError
            If input types or shapes are incompatible.
        """
        if not isinstance(y_true, TensorLike):
            raise ValueError(f"y_true must be a tensor, got {type(y_true)}")

        if metric_params is None:
            metric_params = MetricParams()

        # Ensure 2D y_true for consistent handling
        y_true = y_true.unsqueeze(-1) if y_true.ndim == 1 else y_true

        # Initialize CRPS metric (computes mean by default)
        crps_metric = ContinuousRankedProbabilityScore()
        crps_metric.to(y_true.device)

        # Handle different prediction types
        if isinstance(y_pred, DistributionLike):
            # Distribution case: sample from it
            samples = rearrange(
                y_pred.sample(torch.Size((metric_params.n_samples,))),
                "s b ... -> b ... s",
            )
            if samples.shape[:-1] != y_true.shape:
                raise ValueError(
                    f"Sampled predictions shape {samples.shape[:-1]} (excluding sample "
                    f"dimension) does not match y_true shape {y_true.shape}"
                )
        elif isinstance(y_pred, TensorLike):
            # Tensor case: check dimensions
            if y_pred.dim() == y_true.dim():
                # Deterministic: same shape as y_true
                # CRPS requires at least 2 ensemble members, so duplicate the prediction
                samples = y_pred.unsqueeze(-1).repeat_interleave(2, dim=-1)
            elif y_pred.dim() == y_true.dim() + 1:
                # Probabilistic: already has sample dimension at end
                samples = y_pred
                if samples.shape[:-1] != y_true.shape:
                    raise ValueError(
                        f"y_pred shape {samples.shape[:-1]} (excluding last dimension) "
                        f"does not match y_true shape {y_true.shape}"
                    )
            else:
                raise ValueError(
                    f"y_pred dimensions ({y_pred.dim()}) incompatible with y_true "
                    f"dimensions ({y_true.dim()}). Expected same dimensions or "
                    f"y_true.dim() + 1"
                )
        else:
            raise ValueError(
                f"y_pred must be a tensor or distribution, got {type(y_pred)}"
            )

        # Flatten batch and target dimensions
        samples_flat = samples.flatten(end_dim=-2)  # (batch * targets, n_samples)
        y_true_flat = y_true.flatten()  # (batch * targets,)

        # ContinuousRankedProbabilityScore computes mean by default
        return crps_metric(samples_flat, y_true_flat)


class MSLLMetric(ProbabilisticMetric):
    """Mean Standardized Log Loss (MSLL) metric.

    MSLL evaluates the quality of probabilistic predictions by measuring the
    log-likelihood of the true values under the predictive distribution,
    standardized by the log-likelihood under the trivial model (i.e., predictive
    normal distribution parameterized with the data mean and variance).

    If no training data is supplied, the mean log loss is computed.

    Lower MSLL values indicate better predictive performance.

    Note: This metric requires probabilistic predictions. Standardization
    assumes that the predictive distribution is Gaussian.

    Attributes
    ----------
    name: str
        Display name for the metric.
    maximize: bool
        Whether higher values are better. False for MSLL (lower is better).
    """

    name: str = "msll"
    maximize: bool = False

    def __call__(
        self,
        y_pred: OutputLike,
        y_true: TensorLike,
        metric_params: MetricParams | None = None,
    ) -> TensorLike:
        """Calculate MSLL metric.

        If no training data is provided in `metric_params.y_train`, the mean log loss
        is computed without standardization.

        Parameters
        ----------
        y_pred: OutputLike
            Predicted outputs. Must be a distribution.
        y_true: TensorLike
            True target values.
        metric_params: MetricParams
            Metric parameters including: y_train and reduction.

        Returns
        -------
        TensorLike
            Mean Standardized Log Loss (MSLL) score.

        Raises
        ------
        ValueError
            If y_pred is not a distribution.
        """
        if not isinstance(y_pred, DistributionLike):
            raise ValueError(
                f"MSLL metric requires probabilistic predictions, got {type(y_pred)}. "
            )

        if not isinstance(y_true, TensorLike):
            raise ValueError(f"y_true must be a tensor, got {type(y_true)}")

        if metric_params is None:
            metric_params = MetricParams()

        # Ensure 2D y_true for consistent handling
        y_true = y_true.unsqueeze(-1) if y_true.ndim == 1 else y_true

        # Compute mean negative log likelihood (also by output dimension if have
        # Independent distribution to support 'none' reduction)
        if isinstance(y_pred, Independent):
            model_nll_output = -y_pred.base_dist.log_prob(y_true).mean(dim=0)
            model_nll_total = model_nll_output.mean()
        else:
            model_nll_output = None
            model_nll_total = -y_pred.log_prob(y_true).mean()

        # If no training data, return mean log loss
        if metric_params.y_train is None:
            if metric_params.reduction == "mean":
                return model_nll_total
            if metric_params.reduction == "none":
                if model_nll_output is None:
                    msg = (
                        "Per-output MLL not available for non-Independent "
                        "distributions."
                    )
                    raise ValueError(msg)
                return model_nll_output.reshape(*y_true.shape)
            msg = (
                f"Unknown reduction method: {metric_params.reduction}. "
                "Expected 'mean' or 'none'."
            )
            raise ValueError(msg)

        # Keep original shape for y_train_mean to match y_true shape
        y_train_mean = metric_params.y_train.mean(dim=0, keepdim=True)

        # following GPyTorch implementation, use global variance rather than per task
        # https://github.com/cornellius-gp/gpytorch/blob/c0fb6c64311fdbef2862fd3ba2bd613fbd081e79/gpytorch/metrics/metrics.py#L60
        y_train_var = metric_params.y_train.var()

        # Avoid numerical issues
        y_train_var = torch.clamp(y_train_var, min=1e-6)

        # Compute mean negative log likelihood under trivial Gaussian model
        trivial_nll_output = 0.5 * (
            torch.log(2 * torch.pi * y_train_var)
            + torch.square(y_true - y_train_mean) / (2 * y_train_var)
        ).mean(dim=0)

        # Return mean standardized log loss
        if metric_params.reduction == "mean":
            return model_nll_total - trivial_nll_output.mean()
        if metric_params.reduction == "none":
            if model_nll_output is None:
                msg = "Per-output MLL not available for non-Independent distributions."
                raise ValueError(msg)
            return (model_nll_output - trivial_nll_output).reshape(*y_true.shape)
        msg = (
            f"Unknown reduction method: {metric_params.reduction}. "
            "Expected 'mean' or 'none'."
        )
        raise ValueError(msg)


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

CRPS = CRPSMetric()

MSLL = MSLLMetric()

AVAILABLE_METRICS = {
    "r2": R2,
    "rmse": RMSE,
    "mse": MSE,
    "mae": MAE,
    "crps": CRPS,
    "msll": MSLL,
}


def get_metric(metric: str | Metric) -> Metric:
    """Convert metric specification to a `Metric`.

    Parameters
    ----------
    metric: str | Metric
        The metric specification. Can be:
        - A string shortcut like "r2", "rmse", "mse", "mae"
        - A Metric instance (returned as-is)

    Returns
    -------
    Metric
        The metric.

    Raises
    ------
    ValueError
        If the metric specification is not a string (and registered in
        AVAILABLE_METRICS) or Metric instance.

    """
    # If already a Metric, return as-is
    if isinstance(metric, Metric):
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


def get_metrics(metrics: Sequence[str | Metric]) -> list[Metric]:
    """Convert a list of metric specifications to list of `Metric`s.

    Parameters
    ----------
    metrics: Sequence[str | Metric]
        Sequence of metric specifications.

    Returns
    -------
    list[Metric]
        List of metrics.
    """
    return [get_metric(m) for m in metrics]
