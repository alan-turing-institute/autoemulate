from typing import Any

import numpy as np
import torchmetrics
from sklearn.model_selection import BaseCrossValidator
from torch.utils.data import DataLoader, Dataset, Subset

from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.types import (
    DistributionLike,
    InputLike,
    ModelConfig,
    OutputLike,
    TensorLike,
)


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
    y_true: InputLike, y_pred: OutputLike, metric: type[torchmetrics.Metric]
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

    metric_instance = metric()
    _update(y_true, y_pred, metric_instance)
    return metric_instance.compute().item()


def cross_validate(
    cv: BaseCrossValidator,
    dataset: Dataset,
    model: type[Emulator],
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

    Returns
    -------
    dict[str, list[float]]
       Contains r2 and rmse scores computed for each cross validation fold.
    """
    best_model_config: ModelConfig = kwargs
    cv_results = {"r2": [], "rmse": []}
    batch_size = best_model_config.get("batch_size", 16)
    for train_idx, val_idx in cv.split(dataset):  # type: ignore TODO: identify type handling here
        # create train/val data subsets
        # convert idx to list to satisfy type checker
        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())
        train_loader = DataLoader(train_subset, batch_size=batch_size)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        # fit model
        x, y = next(iter(train_loader))
        # TODO: should we set random_state in the models?
        rs = np.random.randint(int(1e5))
        m = model(x, y, random_state=rs, **best_model_config)
        m = model(x, y, **best_model_config)
        m.fit(x, y)

        # evaluate on batches
        r2_metric = torchmetrics.R2Score()
        mse_metric = torchmetrics.MeanSquaredError()
        for x_batch, y_batch in val_loader:
            y_batch_pred = m.predict(x_batch)
            _update(y_batch, y_batch_pred, r2_metric)
            _update(y_batch, y_batch_pred, mse_metric)

        # compute and save results
        r2 = r2_metric.compute().item()
        rmse = np.sqrt(mse_metric.compute().item())
        cv_results["r2"].append(r2)
        cv_results["rmse"].append(rmse)
    return cv_results
