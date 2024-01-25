import logging
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV

from autoemulate.utils import (
    get_model_name,
    get_model_params,
    get_model_param_grid,
    adjust_param_grid,
)


def optimize_params(
    X,
    y,
    cv,
    model,
    search_type="random",
    niter=20,
    param_grid=None,
    n_jobs=None,
    logger=None,
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
        Type of search to perform. Can be "random" or "bayes", "grid" not yet implemented.
    niter : int, default=20
        Number of parameter settings that are sampled. Trades off runtime vs quality of the solution.
        param_grid : dict, default=None
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored. This enables searching over any sequence of parameter
        settings. Parameters names should be prefixed with "model__" to indicate that
        they are parameters of the model.
    n_jobs : int
        Number of jobs to run in parallel.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    searcher : sklearn.model_selection._search.BaseSearchCV or skopt.searchcv.BayesSearchCV
        Searcher instance.
    """
    model_name = get_model_name(model)
    logger.info(f"Performing grid search for {model_name}...")
    param_grid = process_param_grid(model, search_type, param_grid)
    search_type = search_type.lower()

    # random search
    if search_type == "random":
        searcher = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=niter,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
        )
    # Bayes search
    elif search_type == "bayes":
        searcher = BayesSearchCV(
            model,
            param_grid,
            n_iter=niter,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
        )
    elif search_type == "grid":
        raise NotImplementedError("Grid search not available yet.")
    else:
        raise ValueError(f"Invalid search type: {search_type}")

    # run hyperparameter search
    try:
        searcher.fit(X, y)
    except Exception as e:
        logger.info(
            f"Failed to perform hyperparameter search on {get_model_name(model)}"
        )
        logger.info(e)
    logger.info(f"Best parameters for {model_name}: {searcher.best_params_}")

    return searcher.best_estimator_


def process_param_grid(model, search_type, param_grid):
    """Process parameter grid for hyperparameter search.
    Gets the parameter grid for the model and adjusts it to include prefixes
    for pipelines / multioutput estimators.

    Parameters
    ----------
    model : model instance to do hyperparameter search for.
    search_type : str, default="random"
        Type of search to perform. Can be "random" or "bayes", "grid" not yet implemented.
    param_grid : dict, default=None
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored. This enables searching over any sequence of parameter
        settings.

    Returns
    -------
    param_grid : dict
        Adjusted parameter grid.
    """
    # get param_grid if not provided
    if param_grid is None:
        param_grid = get_model_param_grid(model, search_type)
    else:
        param_grid = check_param_grid(param_grid, model)
    # include prefixes for pipelines / multioutput estimators
    param_grid = adjust_param_grid(model, param_grid)
    return param_grid


def check_param_grid(param_grid, model):
    """Checks that the parameter grid is valid.

    Parameters
    ----------
    param_grid : dict, default=None
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored. This enables searching over any sequence of parameter
        settings. Parameters names should be prefixed with "model__" to indicate that
        they are parameters of the model.
    model : sklearn.pipeline.Pipeline
        Model to be optimized.

    Returns
    -------
    param_grid : dict
    """
    if type(param_grid) != dict:
        raise TypeError("param_grid must be a dictionary")
    for key, value in param_grid.items():
        if type(key) != str:
            raise TypeError("param_grid keys must be strings")
        if type(value) != list:
            raise TypeError("param_grid values must be lists")

    inbuilt_grid = get_model_params(model)
    # check that all keys in param_grid are in the inbuilt grid
    for key in param_grid.keys():
        if key not in inbuilt_grid.keys():
            raise ValueError(f"Invalid parameter: {key}")

    return param_grid
