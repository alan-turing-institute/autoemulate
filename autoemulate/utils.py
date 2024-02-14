import os
import random
import warnings
from contextlib import contextmanager

import numpy as np
import torch
from sklearn.base import RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline


@contextmanager
def suppress_convergence_warnings():
    """Context manager to suppress sklearn convergence warnings."""
    # store the current state of the warning filters and environment variable
    original_filters = warnings.filters[:]
    original_env = os.environ.get("PYTHONWARNINGS")

    # set the desired warning behavior
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    # ensures that warnings are also not shown in subprocesses
    os.environ["PYTHONWARNINGS"] = "ignore"

    try:
        yield
    finally:
        # revert the warning filters and environment variable to their original state
        warnings.filters = original_filters
        if original_env is not None:
            os.environ["PYTHONWARNINGS"] = original_env
        else:
            del os.environ["PYTHONWARNINGS"]


def get_model_name(model):
    """Get the name of the base model.

    This function handles standalone models, models wrapped in a MultiOutputRegressor,
    and models inside a pipeline (possibly wrapped in a MultiOutputRegressor).

    Parameters
    ----------
    model : model instance or Pipeline
        The model or pipeline from which to retrieve the base model name.

    Returns
    -------
    str
        Name of the base model.
    """
    # If the model is a Pipeline
    if isinstance(model, Pipeline):
        # Extract the model step
        step = model.named_steps["model"]

        # If the model step is a MultiOutputRegressor, get the estimator
        if isinstance(step, MultiOutputRegressor):
            return type(step.estimator).__name__
        else:
            return type(step).__name__

    # If the model is a MultiOutputRegressor but not in a pipeline
    elif isinstance(model, MultiOutputRegressor):
        return type(model.estimator).__name__

    # Otherwise, it's a standalone model
    else:
        return type(model).__name__


def get_model_params(model):
    """Get the parameters of the base model, which are not prefixed with `model__` or `estimator__`.

    This function handles standalone models, models wrapped in a MultiOutputRegressor,
    and models inside a pipeline (possibly wrapped in a MultiOutputRegressor).

    Parameters
    ----------
    model : model instance or Pipeline and/or MultiOutputRegressor
        The model or pipeline from which to retrieve the base model parameters.

    Returns
    -------
    dict
        Parameters of the base model.
    """

    if isinstance(model, Pipeline):
        step = model.named_steps["model"]

        if isinstance(step, MultiOutputRegressor):
            return step.estimator.get_params()
        else:
            return step.get_params()

    # If the model is a MultiOutputRegressor but not in a pipeline
    elif isinstance(model, MultiOutputRegressor):
        return model.estimator.get_params()

    # Otherwise, it's a standalone model
    else:
        return model.get_params()


def get_model_param_space(model, search_type="random"):
    """Get the parameter grid of the base model. This is used for hyperparameter search.

    This function handles standalone models, models wrapped in a MultiOutputRegressor,
    and models inside a pipeline (possibly wrapped in a MultiOutputRegressor).

    Parameters
    ----------
    model : model instance or Pipeline and/or MultiOutputRegressor
        The model or pipeline from which to retrieve the base model parameter grid.
    search_type : str
        The type of hyperparameter search to be performed. Can be "random" or "bayes".
        Default is "random".

    Returns
    -------
    dict
        Parameter grid of the base model.
    """
    if isinstance(model, Pipeline):
        step = model.named_steps["model"]

        if isinstance(step, MultiOutputRegressor):
            return step.estimator.get_grid_params(search_type)
        else:
            return step.get_grid_params(search_type)

    # If the model is a MultiOutputRegressor but not in a pipeline
    elif isinstance(model, MultiOutputRegressor):
        return model.estimator.get_grid_params(search_type)

    # Otherwise, it's a standalone model
    else:
        return model.get_grid_params(search_type)


def adjust_param_space(model, param_space):
    """Adjusts param grid to be compatible with the model.
    Adds `model__` if model is a pipeline and
    `estimator__` if model is a MultiOutputRegressor. Or `model__estimator__` if both,
    or returns same param grid if model is a standalone model.

    Parameters
    ----------
    model : model instance or Pipeline
        The pipeline, model or MultiOutputRegressor to which the param grid should be adjusted.
    param_space : dict
        The param grid to be adjusted. This is a dictionary with the parameter names as keys
        which would work for GridSearchCv if the model wouldn't be wrapped in a pipeline or
        MultiOutputRegressor.

    Returns
    -------
    param_space : dict
        Adjusted param grid.
    """
    if isinstance(model, Pipeline):
        step = model.named_steps["model"]

        if isinstance(step, MultiOutputRegressor):
            prefix = "model__estimator__"
        else:
            prefix = "model__"
    elif isinstance(model, MultiOutputRegressor):
        prefix = "estimator__"
    # if model isn't wrapped in anything return param_space as is
    elif isinstance(model, RegressorMixin):
        return param_space

    return add_prefix_to_param_space(param_space, prefix)


def add_prefix_to_param_space(param_space, prefix):
    """Adds a prefix to all keys in a parameter grid.

    Works for three types of param_spaces:

    - when param_space is a dict (standard case)
    - when param_space is a list of dicts (when we only want
      to iterate over certain parameter combinations, like in RBF)
    - when param_space contains tuples of (dict, int) (when we want
      to iterate a certain number of times over a parameter subspace
      (only in BayesSearchCV). This can be used to prevent bayes search
      from iterating many times using the same parameters.

    Parameters
    ----------
    param_space : dict or list of dicts
        The parameter grid to which the prefix will be added.
    prefix : str
        The prefix to be added to each parameter name in the grid.

    Returns
    -------
    dict or list of dicts
        The parameter grid with the prefix added to each key.
    """
    if isinstance(param_space, list):
        new_param_space = []
        for param in param_space:
            # Handle tuple (dict, int) used in BayesSearchCV
            if isinstance(param, tuple):
                # Reconstruct the tuple with the modified dictionary
                dict_with_prefix = add_prefix_to_single_grid(param[0], prefix)
                new_param_space.append((dict_with_prefix,) + param[1:])
            elif isinstance(param, dict):
                # Add prefix to the dictionary
                new_param_space.append(add_prefix_to_single_grid(param, prefix))
        return new_param_space
    else:
        # If param_space is a single dictionary, add the prefix directly
        return add_prefix_to_single_grid(param_space, prefix)


def add_prefix_to_single_grid(grid, prefix):
    """Adds a prefix to all keys in a single parameter grid dictionary."""
    return {prefix + key: value for key, value in grid.items()}


def normalise_y(y):
    """Normalize the target values y.

    Parameters
    ----------
    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        The target values (real numbers).

    Returns
    -------
    y_norm : array-like, shape (n_samples,) or (n_samples, n_outputs)
        The normalized target values.
    y_mean : array-like, shape (n_outputs,)
        The mean of the target values.
    y_std : array-like, shape (n_outputs,)
        The standard deviation of the target values.

    """
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    return (y - y_mean) / y_std, y_mean, y_std


def denormalise_y(y_pred, y_mean, y_std):
    """Denormalize the predicted values.

    Parameters
    ----------
    y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
        The predicted target values.
    y_mean : array-like, shape (n_outputs,)
        The mean of the target values.
    y_std : array-like, shape (n_outputs,)
        The standard deviation of the target values.

    Returns
    -------
    y_pred_denorm : array-like, shape (n_samples,) or (n_samples, n_outputs)
        The denormalized predicted target values.

    """
    return y_pred * y_std + y_mean


def get_mean_scores(scores_df, metric):
    """Get the mean scores for each model and metric.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame with columns "model", "metric", "fold", "score".
    metric : str
        The metric for which to calculate the mean score. Currently supported are "r2" and "rmse".

    Returns
    -------
    mean_scores_df : pandas.DataFrame
        DataFrame with columns "model", "metric", "mean_score".
    """

    # check if metric is in scores_df metric column
    if metric not in scores_df["metric"].unique():
        raise ValueError(
            f"Metric {metric} not found. Available metrics are: {scores_df['metric'].unique()}"
        )

    if metric == "r2":
        asc = False
    elif metric == "rmse":
        asc = True
    else:
        raise RuntimeError(f"Metric {metric} not supported.")

    means_df = (
        scores_df.groupby(["model", "metric"])["score"]
        .mean()
        .unstack()
        .reset_index()
        .sort_values(by=metric, ascending=asc)
        .rename_axis(None, axis=1)
        .reset_index(drop=True)
    )

    return means_df


def get_model_scores(scores_df, model_name):
    """
    Get the scores for a specific model.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame with columns "model", "metric", "fold", "score".
    model_name : str
        The name of the model for which to retrieve the scores.

    Returns
    -------
    model_scores : pandas.DataFrame
        DataFrame with columns "fold", "metric", "score".
    """
    model_scores = scores_df[scores_df["model"] == model_name].pivot(
        index="fold", columns="metric", values="score"
    )

    return model_scores


def set_random_seed(seed: int, deterministic: bool = False):
    """Set random seed for Python, Numpy and PyTorch.

    Parameters
    ----------
    seed : int
        The random seed to use.
    deterministic : bool
        Use "deterministic" algorithms in PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
