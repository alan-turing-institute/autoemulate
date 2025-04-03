from typing import Callable

from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from torch.utils.data import DataLoader, Dataset, Subset

from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.types import DistributionLike, TensorLike


def evaluate(y_true: TensorLike, y_pred: TensorLike, score_f: Callable):
    """
    Evaluate Emulator prediction performance using a `score_f` metric.

    Parameters
    ----------
    y_true: TensorLike
        Ground truth target values.
    y_pred: TensorLike
        Predicted target values, as returned by an Emulator.
    score_f: Callable
        Function that takes in ground truth and predicted target values and
        returns a measure of performance (e.g., r2, rmse).

    Returns
    -------
    float
    """
    # handle types
    if isinstance(y_pred, TensorLike):
        score = score_f(y_true, y_pred.detach().numpy())
    elif isinstance(y_pred, DistributionLike):
        score = score_f(y_true, y_pred.mean.detach().numpy())
    elif (
        isinstance(y_pred, tuple)
        and len(y_pred) == 2
        and all(isinstance(item, TensorLike) for item in y_pred)
    ):
        score = score_f(y_true, y_pred[0].detach().numpy())
    else:
        raise ValueError(f"Score not implmented for {type(y_pred)}")
    return score


def cross_validate(cv: BaseCrossValidator, dataset: Dataset, model: Emulator):
    """
    Perform cross validation using the given `cv` strategy.

    Parameters
    ----------
    cv: BaseCrossValidator
        Provides split method that returns train/val Dataset indices using a
        specified cross-validation strategy (e.g., KFold, LeaveOneOut).
    dataset: Dataset
        The data to split.
    model: Emulator
        An instance of an Emulator subclass.

    Returns
    -------
    dict[str, list[float]]
       Contains r2 and rmse scores computed for each cross validation fold.
    """
    cv_results = {"r2": [], "rmse": []}
    for train_idx, val_idx in cv.split(dataset):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset)
        val_loader = DataLoader(val_subset)
        # score and save results
        val_x, val_y = next(iter(val_loader))
        model.fit(train_loader)
        y_pred = model.predict(val_x)
        r2 = evaluate(val_y, y_pred, r2_score)
        rmse = evaluate(val_y, y_pred, root_mean_squared_error)
        cv_results["r2"].append(r2)
        cv_results["rmse"].append(rmse)
    return cv_results
