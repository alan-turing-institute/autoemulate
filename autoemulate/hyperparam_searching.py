import logging
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV

from autoemulate.utils import (
    get_model_name,
    get_model_params,
    get_model_param_grid,
    adjust_param_grid,
)

from autoemulate.emulators import RandomForest
import numpy as np

# import cv
from sklearn.model_selection import KFold


def search(
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

    # get param_grid
    if param_grid is None:
        param_grid = get_model_param_grid(model, search_type)
    else:
        param_grid = check_param_grid(param_grid, model)
    # adjust param_grid to include prefixes
    param_grid = adjust_param_grid(model, param_grid)

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

    searcher.fit(X, y)
    logger.info(f"Best parameters for {model_name}: {searcher.best_params_}")

    return searcher


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


def optimize_params(X, y, cv, model, search_type, niter, n_jobs, logger):
    """Runs hyperparameter search and returns model with best hyperparameters."""
    try:
        searcher = search(
            X=X,
            y=y,
            cv=cv,
            model=model,
            search_type=search_type,
            niter=niter,
            n_jobs=n_jobs,
            logger=logger,
        )
    except Exception as e:
        logger.info(
            f"Failed to perform hyperparameter search on {get_model_name(model)}"
        )
        logger.info(e)

    return searcher.best_estimator_


# if __name__ == "__main__":
#     X = np.random.rand(100, 10)
#     y = np.random.rand(100, 2)
#     cv = KFold(5)
#     model = RandomForest()
#     search_type = "random"
#     niter = 20
#     n_jobs = 1
#     logger = logging.getLogger(__name__)
#     best = optimize_params(X, y, cv, model, search_type, niter, n_jobs, logger)
#     print(best.get_params())
