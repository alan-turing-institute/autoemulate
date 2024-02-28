import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split

from autoemulate.utils import get_model_name


def _run_cv(X, y, cv, model, model_name, metrics, n_jobs, logger):
    """Runs cross-validation on a model.

    Parameters
    ----------
        X : array-like
            Features.
        y : array-like
            Target variable.
        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
        model : scikit-learn model
            Model to cross-validate.
        model_name : str
            Name of the model.
        metrics : list
            List of metrics to use for cross-validation.
        n_jobs : int
            Number of jobs to run in parallel.
        logger : logging.Logger
            Logger object.

    Returns
    -------
        fitted_model : scikit-learn model
            Fitted model.
        cv_results : dict
            Results of the cross-validation.
    """
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


def _update_scores_df(scores_df, model_name, cv_results):
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
                    "model": model_name,
                    "metric": key.split("test_", 1)[1],
                    "fold": fold,
                    "score": score,
                }
    return scores_df
