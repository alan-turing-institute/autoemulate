import logging
import re

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

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


def _sum_cv(cv_result):
    """summarises the cv result for a single model

    Parameters
    ----------
        cv_result : dict
            Results of the cross-validation, output of scikit-learn's `cross_validate`.

    Returns
    -------
        cv_sum : pandas.DataFrame
            DataFrame with columns "fold" plus one column per metric.
    """
    # get test scores
    cv_sum = {
        k.split("test_", 1)[1]: v for k, v in cv_result.items() if k.startswith("test_")
    }
    cv_sum = pd.DataFrame(cv_sum).reset_index(names=["fold"])
    return cv_sum


def _sum_cvs(cv_results, sort_by="r2"):
    """summarises the cv results for all models, averaging over folds within each model

    Parameters
    ----------
        cv_results : dict
            model_name: cv_result, where cv_result is the output of scikit-learn's `cross_validate`.

    Returns
    -------
        cv_sum : pandas.DataFrame
            DataFrame with columns "model", "short", and one column per metric, showing
            the mean score across all folds for each model.
    """
    cv_all = []

    # concat all cv result df's
    for model_name, cv_result in cv_results.items():
        df = _sum_cv(cv_result)
        df.insert(0, "short", "".join(re.findall(r"[A-Z]", model_name)).lower())
        df.insert(0, "model", model_name)
        cv_all.append(df)

    # mean over folds
    cv_all = (
        pd.concat(cv_all, axis=0)
        .groupby(["model", "short"])
        .mean()
        .reset_index()
        .drop(columns=["fold"])
    )

    # sort by the metric
    if sort_by not in cv_all.columns:
        raise ValueError(
            f"Metric {sort_by} not found. Available metrics are: {cv_all.columns.drop('model').drop('short').to_list()}"
        )

    # TODO: make this work properly for different metrics
    if sort_by == "r2":
        asc = False
    else:
        asc = True

    cv_all = cv_all.sort_values(by=sort_by, ascending=asc).reset_index(drop=True)

    return cv_all
