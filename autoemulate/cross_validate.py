from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split

from autoemulate.types import ArrayLike
from autoemulate.types import MatrixLike
from autoemulate.types import Union
from autoemulate.utils import get_model_name

if TYPE_CHECKING:
    from logging import Logger
    from .types import Iterable
    from sklearn.model_selection import BaseCrossValidator
    from sklearn.model_selection import BaseShuffleSplit
    from sklearn.pipeline import Pipeline


def run_cv(
    X: MatrixLike,
    y: Union[MatrixLike, ArrayLike],
    cv: Union[int, BaseCrossValidator, Iterable, BaseShuffleSplit],
    model: Pipeline,  # TODO: Verify that this is correct
    metrics: list,
    n_jobs: int,
    logger: Logger,
):
    """Runs cross-validation on a model.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of supervised learning.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    model : sklearn.pipeline.Pipeline
        Model to cross-validate.
    metrics : list
        List of metrics to use for cross-validation.
    n_jobs : int
        Number of jobs to run in parallel.
    logger : logging.Logger
        Logger object.

    Returns
    -------
    fitted_model : sklearn.pipeline.Pipeline
        Fitted model.
    cv_results : dict
        Results of the cross-validation.
    """
    model_name = get_model_name(model)

    # The metrics we want to use for cross-validation
    scorers = {metric.__name__: make_scorer(metric) for metric in metrics}

    logger.info(f"Cross-validating {model_name}...")
    logger.info(f"Parameters: {model.named_steps['model'].get_params()}")

    try:
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scorers,
            n_jobs=n_jobs,
            return_estimator=True,
            return_indices=True,
        )

    except Exception as e:
        logger.error(f"Failed to cross-validate {model_name}")
        logger.error(e)

    # refit the model on the whole dataset
    fitted_model = model.fit(X, y)

    return fitted_model, cv_results


# TODO for update_scores_df: suggestion, rename model to model_name here to not confuse with other references to the model object
def update_scores_df(scores_df: pd.DataFrame, model: str, cv_results: dict) -> None:
    """Updates the scores dataframe with the results of the cross-validation.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame with columns "model", "metric", "fold", "score".
    model_name : str
        Name of the model.
    cv_results : dict
        Results of the cross-validation.

    Returns
    -------
    None
        Modifies the self.scores_df DataFrame in-place.

    """
    # Gather scores from each metric
    # Initialise scores dataframe
    for key in cv_results.keys():
        if key.startswith("test_"):
            for fold, score in enumerate(cv_results[key]):
                scores_df.loc[len(scores_df.index)] = {
                    "model": get_model_name(model),
                    "metric": key.split("test_", 1)[1],
                    "fold": fold,
                    "score": score,
                }
    return scores_df
