from typing import Callable

from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from torch.utils.data import DataLoader, Dataset, Subset

from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.types import DistributionLike, TensorLike


def evaluate(y_true: TensorLike, y_pred: TensorLike, score_f: Callable):
    """
    Evaluate Emulator prediction performance using `score_f` metric.

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
