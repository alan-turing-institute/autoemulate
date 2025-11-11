import inspect
import logging

import torch
from sklearn.model_selection import BaseCrossValidator
from torch.distributions import Transform
from torch.utils.data import Dataset, Subset

from autoemulate.core.device import get_torch_device, move_tensors_to_device
from autoemulate.core.metrics import R2, Metric, get_metrics
from autoemulate.core.types import (
    DeviceLike,
    ModelParams,
    OutputLike,
    TensorLike,
    TransformedEmulatorParams,
)
from autoemulate.data.utils import ConversionMixin, set_random_seed
from autoemulate.emulators.base import Emulator
from autoemulate.emulators.transformed.base import TransformedEmulator

logger = logging.getLogger("autoemulate")


def evaluate(
    y_pred: OutputLike,
    y_true: TensorLike,
    metric: Metric = R2,
    n_samples: int = 1000,
    metric_kwargs: dict | None = None,
) -> float:
    """
    Evaluate Emulator prediction performance using a `torchmetrics.Metric`.

    Parameters
    ----------
    y_pred: OutputLike
        Predicted target values, as returned by an Emulator.
    y_true: TensorLike
        Ground truth target values.
    metric: Metric
        Metric to use for evaluation. Defaults to R2.
    n_samples: int
        Number of samples to generate to predict mean when y_pred does not have a mean
        directly available. Defaults to 1000.
    metric_kwargs: dict | None
        Additional keyword arguments to pass to the metric.

    Returns
    -------
    float
    """
    metric_kwargs = metric_kwargs or {}
    return metric(y_pred, y_true, n_samples=n_samples, **metric_kwargs).item()


def cross_validate(
    cv: BaseCrossValidator,
    dataset: Dataset,
    model: type[Emulator],
    model_params: ModelParams,
    transformed_emulator_params: None | TransformedEmulatorParams = None,
    x_transforms: list[Transform] | None = None,
    y_transforms: list[Transform] | None = None,
    device: DeviceLike = "cpu",
    random_seed: int | None = None,
    metrics: list[Metric] | None = None,
):
    """
    Cross validate model performance using the given `cv` strategy.

    Parameters
    ----------
    cv: BaseCrossValidator
        Provides split method that returns train/val Dataset indices using the
        specified cross-validation strategy (e.g., KFold, LeaveOneOut).
    dataset: Dataset
        The data to use for model training and validation.
    model: Emulator
        An instance of an Emulator subclass.
    model_params: ModelParams
        Model parameters to be used to construct model upon initialization. Passing an
        empty dictionary `{}` will use default parameters.
    transformed_emulator_params: None | TransformedEmulatorParams
        Parameters for the transformed emulator. Defaults to None.
    device: DeviceLike
        The device to use for model training and evaluation.
    random_seed: int | None
        Optional random seed for reproducibility.
    metrics: list[TorchMetrics] | None
        List of metrics to compute. If None, uses r2 and rmse.

    Returns
    -------
    dict[str, list[float]]
       Contains scores for each metric computed for each cross validation fold.
    """
    transformed_emulator_params = transformed_emulator_params or {}
    x_transforms = x_transforms or []
    y_transforms = y_transforms or []

    # Setup metrics
    if metrics is None:
        metrics = get_metrics(["r2", "rmse"])

    cv_results = {metric.name: [] for metric in metrics}
    device = get_torch_device(device)

    logger.debug("Cross-validation parameters: %s", cv)
    for i, (train_idx, val_idx) in enumerate(cv.split(dataset)):  # type: ignore TODO: identify type handling here
        logger.debug(
            "Cross-validation split %d: %d train samples, %d validation samples",
            i,
            len(train_idx),
            len(val_idx),
        )

        # create train/val data subsets
        # convert idx to list to satisfy type checker
        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())

        # Handle random seed for reproducibility
        if random_seed is not None:
            set_random_seed(seed=random_seed)
        model_init_params = inspect.signature(model).parameters
        model_params = dict(model_params)
        if "random_seed" in model_init_params:
            model_params["random_seed"] = random_seed

        # Convert dataloader to tensors to pass to model
        x, y = ConversionMixin._convert_to_tensors(train_subset)
        x_val, y_val = ConversionMixin._convert_to_tensors(val_subset)

        transformed_emulator = TransformedEmulator(
            x,
            y,
            model=model,
            x_transforms=x_transforms,
            y_transforms=y_transforms,
            device=device,
            **model_params,
            **transformed_emulator_params,
        )
        transformed_emulator.fit(x, y)

        # compute and save results
        y_pred = transformed_emulator.predict(x_val)
        for metric in metrics:
            score = evaluate(y_pred, y_val, metric)
            cv_results[metric.name].append(score)
    return cv_results


def bootstrap(
    model: Emulator,
    x: TensorLike,
    y: TensorLike,
    n_bootstraps: int | None = 100,
    n_samples: int = 1000,
    device: str | torch.device = "cpu",
    metrics: list[Metric] | None = None,
    metric_kwargs: dict[str, dict] | None = None,
) -> dict[Metric, tuple[float, float]]:
    """
    Get bootstrap estimates of metrics.

    Parameters
    ----------
    model: Emulator
        An instance of an Emulator subclass.
    x: TensorLike
        Input features for the model.
    y: TensorLike
        Target values corresponding to the input features.
    n_bootstraps: int | None
        Number of bootstrap samples to generate. When None the evaluation uses all
        all given data and returns a single value with no measure of the uncertainty.
        Defaults to 100.
    n_samples: int
        Number of samples to generate to predict mean when emulator does not have a
        mean directly available. Defaults to 1000.
    device: str | torch.device
        The device to use for computations. Default is "cpu".
    metrics: list[MetricConfig] | None
        List of metrics to compute. If None, uses r2 and rmse.
    metric_kwargs: dict[str, dict] | None
        Dictionary mapping metric names to their specific kwargs.
        Example: {"r2": {"multioutput": "uniform_average"}}


    Returns
    -------
    dict[Metric, tuple[float, float]]
        Dictionary mapping metrics to (mean, std) tuples.
    """
    device = get_torch_device(device)
    x, y = move_tensors_to_device(x, y, device=device)

    # Setup metrics
    if metrics is None:
        metrics = get_metrics(["r2", "rmse"])

    metric_kwargs = metric_kwargs or {}

    # If no bootstraps are specified, fall back to a single evaluation on given data
    if n_bootstraps is None:
        y_pred = model.predict(x)
        results = {}
        for metric in metrics:
            score = evaluate(y_pred, y, metric=metric, n_samples=n_samples)
            results[metric] = (score, float("nan"))
        return results

    # Initialize score tensors for each metric
    metric_scores = {
        metric.name: torch.empty(n_bootstraps, device=device) for metric in metrics
    }

    for i in range(n_bootstraps):
        # Bootstrap sample indices with replacement
        idxs = torch.randint(0, x.shape[0], size=(x.shape[0],), device=device)

        # Get bootstrap sample
        x_bootstrap = x[idxs]
        y_bootstrap = y[idxs]

        # Make predictions
        y_pred = model.predict(x_bootstrap)

        # Compute metrics for this bootstrap sample
        for metric in metrics:
            kwargs = metric_kwargs.get(metric.name, {})
            metric_scores[metric.name][i] = evaluate(
                y_pred,
                y_bootstrap,
                metric=metric,
                n_samples=n_samples,
                metric_kwargs=kwargs,
            )

    # Return mean and std for each metric
    return {
        metric: (
            metric_scores[metric.name].mean().item(),
            metric_scores[metric.name].std().item(),
        )
        for metric in metrics
    }
