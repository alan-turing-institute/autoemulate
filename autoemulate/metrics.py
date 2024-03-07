import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error


def rmse(y_true, y_pred):
    """Returns the root mean squared error.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_outputs)
        Simulation output.
    y_pred : array-like, shape (n_samples, n_outputs)
        Emulator output.
    """
    return root_mean_squared_error(y_true, y_pred)


def r2(y_true, y_pred):
    """Returns the R^2 score.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_outputs)
        Simulation output.
    y_pred : array-like, shape (n_samples, n_outputs)
        Emulator output.
    """
    return r2_score(y_true, y_pred)


#: A dictionary of available metrics.
METRIC_REGISTRY = {
    "rmse": rmse,
    "r2": r2,
}
