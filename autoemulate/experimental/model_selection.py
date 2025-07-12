import inspect
import logging
from typing import Any

import numpy as np
import torch
import torchmetrics
from sklearn.model_selection import BaseCrossValidator
from torch.utils.data import DataLoader, Dataset, Subset

from autoemulate.experimental.data.utils import ConversionMixin, set_random_seed
from autoemulate.experimental.device import get_torch_device, move_tensors_to_device
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.emulators.transformed.base import TransformedEmulator
from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.types import (
    DeviceLike,
    DistributionLike,
    InputLike,
    ModelConfig,
    OutputLike,
    TensorLike,
)

logger = logging.getLogger("autoemulate")


def _update(
    y_true: InputLike,
    y_pred: OutputLike,
    metric: torchmetrics.Metric,
):
    # handle types
    if isinstance(y_pred, TensorLike):
        metric.update(y_true, y_pred)
    elif isinstance(y_pred, DistributionLike):
        metric.update(y_true, y_pred.mean)
    elif (
        isinstance(y_pred, tuple)
        and len(y_pred) == 2
        and all(isinstance(item, TensorLike) for item in y_pred)
    ):
        metric.update(y_true, y_pred)
    else:
        raise ValueError(f"Metric not implmented for {type(y_pred)}")


def evaluate(
    y_true: InputLike,
    y_pred: OutputLike,
    metric: type[torchmetrics.Metric],
    device: DeviceLike,
) -> float:
    """
    Evaluate Emulator prediction performance using a `torchmetrics.Metric`.

    Parameters
    ----------
    y_true: InputLike
        Ground truth target values.
    y_pred: OutputLike
        Predicted target values, as returned by an Emulator.
    metric: type[Metric]
        A torchmetrics metric to compute the score.

    Returns
    -------
    float
    """

    metric_instance = metric().to(device)
    _update(y_true, y_pred, metric_instance)
    return metric_instance.compute().item()


def cross_validate(  # noqa: PLR0913
    cv: BaseCrossValidator,
    dataset: Dataset,
    model: type[Emulator],
    x_transforms: list[AutoEmulateTransform] | None = None,
    y_transforms: list[AutoEmulateTransform] | None = None,
    device: DeviceLike = "cpu",
    random_seed: int | None = None,
    **kwargs: Any,
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
    device: DeviceLike
        The device to use for model training and evaluation.
    random_seed: int | None
        Optional random seed for reproducibility.
    Returns
    -------
    dict[str, list[float]]
       Contains r2 and rmse scores computed for each cross validation fold.
    """
    best_model_config: ModelConfig = kwargs
    x_transforms = x_transforms or []
    y_transforms = y_transforms or []
    cv_results = {"r2": [], "rmse": []}
    batch_size = best_model_config.get("batch_size", 16)
    device = get_torch_device(device)

    logger.debug("Cross-validation configuration: %s", cv)
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
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        # Handle random seed for reproducibility
        if random_seed is not None:
            set_random_seed(seed=random_seed)
        model_init_params = inspect.signature(model).parameters
        model_kwargs = dict(best_model_config)
        if "random_seed" in model_init_params:
            model_kwargs["random_seed"] = random_seed

        # Convert dataloader to tensors to pass to model
        x, y = ConversionMixin._convert_to_tensors(train_subset)

        transformed_emulator = TransformedEmulator(
            x,
            y,
            model=model,
            x_transforms=x_transforms,
            y_transforms=y_transforms,
            device=device,
            **model_kwargs,
        )
        transformed_emulator.fit(x, y)

        # evaluate on batches
        r2_metric = torchmetrics.R2Score().to(device)
        mse_metric = torchmetrics.MeanSquaredError().to(device)
        for x_b, y_b in val_loader:
            x_b_device, y_b_device = move_tensors_to_device(x_b, y_b, device=device)
            y_batch_pred = transformed_emulator.predict(x_b_device)
            _update(y_b_device, y_batch_pred, r2_metric)
            _update(y_b_device, y_batch_pred, mse_metric)

        # compute and save results
        r2 = r2_metric.compute().item()
        rmse = np.sqrt(mse_metric.compute().item())
        cv_results["r2"].append(r2)
        cv_results["rmse"].append(rmse)
    return cv_results


def bootstrap(
    model: Emulator,
    x: TensorLike,
    y: TensorLike,
    n_bootstraps: int = 100,
    device: str | torch.device = "cpu",
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Gets bootstrap estimates of R2 and RMSE

    Parameters
    ----------
    model : Emulator
        An instance of an Emulator subclass.
    x : TensorLike
        Input features for the model.
    y : TensorLike
        Target values corresponding to the input features.
    n_bootstraps : int=100
        Number of bootstrap samples to generate.
    device : str | torch.device, default="cpu"
        The device to use for computations (default is "cpu").

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        ((r2_mean, r2_std), (rmse_mean, rmse_std))
    """
    device = get_torch_device(device)
    x, y = move_tensors_to_device(x, y, device=device)

    r2_scores = torch.empty(n_bootstraps, device=device)
    rmse_scores = torch.empty(n_bootstraps, device=device)
    for i in range(n_bootstraps):
        # Bootstrap sample indices with replacement
        idxs = torch.randint(0, x.shape[0], size=(x.shape[0],), device=device)

        # Get bootstrap sample
        x_bootstrap = x[idxs]
        y_bootstrap = y[idxs]

        # Make predictions
        y_pred = model.predict(x_bootstrap)

        # Compute metrics for this bootstrap sample
        r2_scores[i] = evaluate(y_bootstrap, y_pred, torchmetrics.R2Score, device)
        mse_score = evaluate(y_bootstrap, y_pred, torchmetrics.MeanSquaredError, device)
        rmse_scores[i] = mse_score**0.5

    # Return mean and std
    return (
        (r2_scores.mean().item(), r2_scores.std().item()),
        (rmse_scores.mean().item(), rmse_scores.std().item()),
    )
