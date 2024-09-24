import logging
import re

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split

from autoemulate.utils import get_model_name
from autoemulate.utils import get_model_params


def _run_cv(X, y, cv, model, metrics, n_jobs=None, logger=None):
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

    # if logger is None, create a new logger
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(f"Cross-validating {get_model_name(model)}...")
    logger.info(f"Parameters: {get_model_params(model)}")

    cv_results = None
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
        # sometimes pickling etc. fails so a model can't be parallelized
        # this is a workaround
        logger.warning(
            f"Parallelized cross-validation failed for {get_model_name(model)}, trying single-threaded cross-validation..."
        )
        try:
            cv_results = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=scorers,
                n_jobs=1,
                return_estimator=True,
                return_indices=True,
            )
        except Exception as e:
            logger.exception(f"Failed to cross-validate {get_model_name(model)}")

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
                    "short": "".join(re.findall(r"[A-Z]", model_name)).lower(),
                    "metric": key.split("test_", 1)[1],
                    "fold": fold,
                    "score": score,
                }
    return scores_df


def _get_mean_scores(scores_df, metric):
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
        scores_df.groupby(["model", "short", "metric"])["score"]
        .mean()
        .unstack()
        .reset_index()
        .sort_values(by=metric, ascending=asc)
        .rename_axis(None, axis=1)
        .reset_index(drop=True)
    )

    return means_df


def _get_model_scores(scores_df, model_name):
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


def _get_cv_results(models, scores_df, model_name=None, sort_by="r2"):
    """Improved print cv results function.

    Parameters
    ----------
    models : list
        List of models.
    scores_df : pandas.DataFrame
        DataFrame with scores for each model, metric, and fold.
    model_name : str, optional
        Specific model name to print scores for. If None, prints best fold for each model.
    sort_by : str, optional
        Metric to sort by. Defaults to "r2".

    Returns
    -------
    out : pandas.DataFrame
        DataFrame with summary of cv results.
    """
    if model_name is not None:
        model_names = [get_model_name(mod) for mod in models]
        if model_name not in model_names:
            raise ValueError(
                f"Model {model_name} not found. Available models: {', '.join(model_names)}"
            )
        df = _get_model_scores(scores_df, model_name)
    else:
        df = _get_mean_scores(scores_df, metric=sort_by)
    return df
