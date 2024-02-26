import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from .types import ArrayLike


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Returns the root mean squared error.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_outputs)
        Simulation output.
    y_pred : array-like, shape (n_samples, n_outputs)
        Emulator output.
    """
    return mean_squared_error(y_true, y_pred, squared=False)


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Returns the R^2 score.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_outputs)
        Simulation output.
    y_pred : array-like, shape (n_samples, n_outputs)
        Emulator output.
    """
    return r2_score(y_true, y_pred)


METRIC_REGISTRY = {
    "rmse": rmse,
    "r2": r2,
}
