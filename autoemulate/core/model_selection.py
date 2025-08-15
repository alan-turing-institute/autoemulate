import inspect
import logging
from functools import partial

import torch
import torchmetrics
from sklearn.model_selection import BaseCrossValidator
from torch.distributions import Transform
from torch.utils.data import Dataset, Subset

from autoemulate.core.device import (
    get_torch_device,
    move_tensors_to_device,
)
from autoemulate.core.types import (
    DeviceLike,
    ModelParams,
    TensorLike,
    TransformedEmulatorParams,
)
from autoemulate.data.utils import ConversionMixin, set_random_seed
from autoemulate.emulators.base import Emulator
from autoemulate.emulators.transformed.base import TransformedEmulator

logger = logging.getLogger("autoemulate")


def r2_metric() -> type[torchmetrics.Metric]:
    """Return a torchmetrics.R2Score metric."""
    return torchmetrics.R2Score


def rmse_metric() -> partial[torchmetrics.Metric]:
    """Return a torchmetrics.MeanSquaredError metric with squared=False."""
    return partial(torchmetrics.MeanSquaredError, squared=False)


def _update(
    y_true: TensorLike,
    y_pred: TensorLike,
    metric: torchmetrics.Metric,
):
    if isinstance(y_pred, TensorLike):
        metric.to(y_pred.device)
        # Assume first dim is a batch dim and flatten remaining for metric calculation
        metric.update(y_pred.flatten(start_dim=1), y_true.flatten(start_dim=1))
    else:
        raise ValueError(f"Metric not implmented for {type(y_pred)}")


def evaluate(
    y_pred: TensorLike,
    y_true: TensorLike,
    metric: (
        type[torchmetrics.Metric] | partial[torchmetrics.Metric]
    ) = torchmetrics.R2Score,
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
    _metric = metric()
    _update(y_true, y_pred, _metric)
    return _metric.compute().item()


def cross_validate(  # noqa: PLR0913
    cv: BaseCrossValidator,
    dataset: Dataset,
    model: type[Emulator],
    model_params: ModelParams,
    transformed_emulator_params: None | TransformedEmulatorParams = None,
    x_transforms: list[Transform] | None = None,
    y_transforms: list[Transform] | None = None,
    device: DeviceLike = "cpu",
    random_seed: int | None = None,
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

    Returns
    -------
    dict[str, list[float]]
       Contains r2 and rmse scores computed for each cross validation fold.
    """
    transformed_emulator_params = transformed_emulator_params or {}
    x_transforms = x_transforms or []
    y_transforms = y_transforms or []
    cv_results = {"r2": [], "rmse": []}
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
        y_pred = transformed_emulator.predict_mean(x_val)
        r2 = evaluate(y_pred, y_val, r2_metric())
        rmse = evaluate(y_pred, y_val, rmse_metric())
        cv_results["r2"].append(r2)
        cv_results["rmse"].append(rmse)
    return cv_results


def bootstrap(  # noqa: PLR0913
    model: Emulator,
    x: TensorLike,
    y: TensorLike,
    n_bootstraps: int | None = 100,
    n_samples: int = 100,
    device: str | torch.device = "cpu",
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Get bootstrap estimates of R2 and RMSE.

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
        mean directly available. Defaults to 100.
    device: str | torch.device
        The device to use for computations. Default is "cpu".

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        ((r2_mean, r2_std), (rmse_mean, rmse_std))
    """
    device = get_torch_device(device)
    x, y = move_tensors_to_device(x, y, device=device)

    # If no bootstraps are specified, fall back to a single evaluation on given data
    if n_bootstraps is None:
        y_pred = model.predict_mean(x, n_samples=n_samples)
        r2_score = evaluate(y_pred, y, r2_metric())
        rmse_score = evaluate(y_pred, y, rmse_metric())
        # Return single score and NaN for std
        return ((r2_score, float("nan")), (rmse_score, float("nan")))

    r2_scores = torch.empty(n_bootstraps, device=device)
    rmse_scores = torch.empty(n_bootstraps, device=device)
    for i in range(n_bootstraps):
        # Bootstrap sample indices with replacement
        idxs = torch.randint(0, x.shape[0], size=(x.shape[0],), device=device)

        # Get bootstrap sample
        x_bootstrap = x[idxs]
        y_bootstrap = y[idxs]

        # Make predictions
        y_pred = model.predict_mean(x_bootstrap, n_samples=n_samples)

        # Compute metrics for this bootstrap sample
        r2_scores[i] = evaluate(y_pred, y_bootstrap, r2_metric())
        rmse_scores[i] = evaluate(y_pred, y_bootstrap, rmse_metric())

    # Return mean and std
    return (
        (r2_scores.mean().item(), r2_scores.std().item()),
        (rmse_scores.mean().item(), rmse_scores.std().item()),
    )
