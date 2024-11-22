import logging

import numpy as np
from sklearn.model_selection import RandomizedSearchCV

from autoemulate.utils import _adjust_param_space
from autoemulate.utils import get_model_name
from autoemulate.utils import get_model_param_space
from autoemulate.utils import get_model_params


def _optimize_params(
    X,
    y,
    cv,
    model,
    search_type="random",
    niter=20,
    param_space=None,
    n_jobs=None,
    logger=None,
    error_score=np.nan,
    verbose=0,
):
    """Performs hyperparameter search for the provided model.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Simulation input samples.
    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        Simulation output.
    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.
    model : model instance to do hyperparameter search for.
    search_type : str, default="random"
        Type of search to perform. Only "random" is supported.
    niter : int, default=20
        Number of parameter settings that are sampled. Trades off runtime vs quality of the solution.
        param_space : dict, default=None
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored. This enables searching over any sequence of parameter
        settings. Parameters names should be prefixed with ``model__`` to indicate that
        they are parameters of the model.
    n_jobs : int
        Number of jobs to run in parallel.
    logger : logging.Logger
        Logger instance
    error_score: 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
    verbose: int
        Verbosity level for the searcher

    Returns
    -------
    Refitted estimator on the whole dataset with best parameters.
    """
    logger.info(f"Performing grid search for {get_model_name(model)}...")
    param_space = _process_param_space(model, search_type, param_space)
    search_type = search_type.lower()

    # random search
    if search_type == "random":
        searcher = RandomizedSearchCV(
            model,
            param_space,
            n_iter=niter,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            error_score=error_score,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Invalid search type: {search_type}")

    # run hyperparameter search
    try:
        searcher.fit(X, y)
    except Exception:
        logger.exception(
            f"Failed to perform hyperparameter search on {get_model_name(model)}"
        )
    logger.info(f"Best parameters for {get_model_name(model)}: {searcher.best_params_}")

    return searcher.best_estimator_


def _process_param_space(model, search_type, param_space):
    """Process parameter grid for hyperparameter search.
    Gets the parameter grid for the model and adjusts it to include prefixes
    for pipelines / multioutput estimators.

    Parameters
    ----------
    model : model instance to do hyperparameter search for.
    search_type : str, default="random"
        Type of search to perform. Only "random" is currently supported.
    param_space : dict, default=None
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored. This enables searching over any sequence of parameter
        settings.

    Returns
    -------
    param_space : dict
        Adjusted parameter grid.
    """
    # get param_space if not provided
    if param_space is None:
        param_space = get_model_param_space(model, search_type)
    else:
        param_space = _check_param_space(param_space, model)
    # include prefixes for pipelines / multioutput estimators
    param_space = _adjust_param_space(model, param_space)
    return param_space


def _check_param_space(param_space, model):
    """Checks that the parameter grid is valid.

    Parameters
    ----------
    param_space : dict, default=None
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored. This enables searching over any sequence of parameter
        settings. Parameters names should be prefixed with ``model__`` to indicate that
        they are parameters of the model.
    model : sklearn.pipeline.Pipeline
        Model to be optimized.

    Returns
    -------
    param_space : dict
    """
    if type(param_space) != dict:
        raise TypeError("param_space must be a dictionary")
    for key, value in param_space.items():
        if type(key) != str:
            raise TypeError("param_space keys must be strings")
        if type(value) != list:
            raise TypeError("param_space values must be lists")

    inbuilt_grid = get_model_params(model)
    # check that all keys in param_space are in the inbuilt grid
    for key in param_space.keys():
        if key not in inbuilt_grid.keys():
            raise ValueError(f"Invalid parameter: {key}")

    return param_space
