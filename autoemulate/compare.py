from autoemulate.metrics import METRIC_REGISTRY
from autoemulate.emulators import MODEL_REGISTRY
from autoemulate.cv import CV_REGISTRY
from autoemulate.logging_config import configure_logging
from autoemulate.plotting import plot_results
from autoemulate.hyperparam_search import HyperparamSearch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.validation import check_X_y
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV


class AutoEmulate:
    def __init__(self):
        """Initializes an AutoEmulate object."""
        self.X = None
        self.y = None
        self.is_set_up = False
        self.scaler = None

    def setup(
        self,
        X,
        y,
        hyperparameter_search=False,
        normalise=True,
        fold_strategy="kfold",
        folds=5,
        n_jobs=None,
        log_to_file=False,
    ):
        """Sets up the automatic emulation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.
        hyperparameter_search : bool
            Whether to perform hyperparameter search over predifined parameter grids.
        normalise : bool, default=False
            Whether to normalise the data before fitting the models. Currently only
            z-transformation using StandardScaler.
        fold_strategy : str
            Cross-validation strategy, currently either "kfold" or "stratified_kfold".
        folds : int
            Number of folds.
        n_jobs : int
            Number of jobs to run in parallel. `None` means 1, `-1` means using all processors.
        log_to_file : bool
            Whether to log to file.
        """
        self.X, self.y = self._check_input(X, y)
        self.models = self._get_models(MODEL_REGISTRY, normalise=normalise)
        self.metrics = [metric for metric in METRIC_REGISTRY.keys()]
        self.cv = CV_REGISTRY[fold_strategy](folds=folds, shuffle=True)
        self.hyperparameter_search = hyperparameter_search
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.logger = configure_logging(log_to_file=log_to_file)
        self.is_set_up = True
        self.cv_results = {}

    def _check_input(self, X, y):
        """Checks and possibly converts the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.

        Returns
        -------
        X_converted : array-like, shape (n_samples, n_features)
            Simulation input.
        y_converted : array-like, shape (n_samples, n_outputs)
            Simulation output.
        """

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True, dtype="float32")
        y = y.astype("float32")  # needed for pytorch models
        return X, y

    def _get_models(self, MODEL_REGISTRY, normalise=True):
        """Get models from REGISTRY and add scaler if normalised.

        Parameters
        ----------
        MODEL_REGISTRY : dict
            Registry of models.
        normalise : bool
            If True, add scaler to models.

        Returns
        -------
        self.models : list
            List of models.

        """
        if normalise:
            self.scaler = StandardScaler()
            models = [
                Pipeline([("scaler", self.scaler), ("model", model())])
                for model in MODEL_REGISTRY.values()
            ]
        else:
            models = [
                Pipeline([("model", model())]) for model in MODEL_REGISTRY.values()
            ]
        return models

    def compare(self):
        """Compares the emulator models on the data. self.setup() must be run first.

        Returns
        -------
        scores_df : pandas.DataFrame
            Dataframe containing the scores for each model, metric and fold.
        """
        if not self.is_set_up:
            raise RuntimeError("Must run setup() before compare()")

        # Freshly initialise scores dataframe when running compare()
        self.scores_df = pd.DataFrame(
            columns=["model", "metric", "fold", "score"]
        ).astype(
            {"model": "object", "metric": "object", "fold": "int64", "score": "float64"}
        )
        # Freshly initialise best parameters for each model
        self.best_params = {}

        for i, model in enumerate(self.models):
            updated_model = (
                self._get_best_hyperparams(i, model)
                if self.hyperparameter_search
                else model
            )
            self.cross_validate(updated_model)

        # returns best model fitted on full data
        return self._get_best_model(metric="r2")

    def cross_validate(self, model):
        """Perform cross-validation on a given model using the specified metrics.

        Parameters
        ----------
            model: A scikit-learn estimator object.

        Class attributes used
        ---------------------
            self.X : array-like, shape (n_samples, n_features)
                Simulation input.
            self.y : array-like, shape (n_samples, n_outputs)
                Simulation output.
            self.cv : scikit-learn cross-validation object
                Cross-validation strategy.
            self.metrics : list of str
                List of metrics to use for cross-validation.
            self.n_jobs : int
                Number of jobs to run in parallel. `None` means 1, `-1` means using all processors.

        Returns
        -------
            scores_df : pandas.DataFrame
                Dataframe containing the scores for each model, metric and fold.

        """

        # Get model name
        model_name = type(model.named_steps["model"]).__name__

        # The metrics we want to use for cross-validation
        scorers = {name: make_scorer(METRIC_REGISTRY[name]) for name in self.metrics}

        self.logger.info(f"Cross-validating {model_name}...")
        self.logger.info(f"Parameters: {model.named_steps['model'].get_params()}")

        # Cross-validate
        cv_results = cross_validate(
            model,
            self.X,
            self.y,
            cv=self.cv,
            scoring=scorers,
            n_jobs=self.n_jobs,
            return_estimator=True,
            return_indices=True,
        )
        # updates pandas dataframe with model cv scores
        self._update_scores_df(model_name, cv_results)
        # save results for plotting etc.
        self.cv_results[model_name] = cv_results

    def _update_scores_df(self, model_name, cv_results):
        """Updates the scores dataframe with the results of the cross-validation.

        Parameters
        ----------
            model_name : str
                Name of the model.
            cv_results : dict
                Results of the cross-validation.

        Returns
        -------
            scores_df : pandas.DataFrame
                Dataframe containing the scores for each model, metric and fold.

        """
        # Gather scores from each metric
        # Initialise scores dataframe
        for key in cv_results.keys():
            if key.startswith("test_"):
                for fold, score in enumerate(cv_results[key]):
                    self.scores_df.loc[len(self.scores_df.index)] = {
                        "model": model_name,
                        "metric": key.split("test_", 1)[1],
                        "fold": fold,
                        "score": score,
                    }

    def _get_best_hyperparams(self, model_index, model):
        """Performs hyperparameter search and updates the model.

        Parameters
        ----------
        model_index : int
            Index of the model in self.models.
        model : scikit-learn estimator object
            Model to perform hyperparameter search on.
        """
        # Perform hyperparameter search and update model
        hyperparam_searcher = HyperparamSearch(
            self.X, self.y, self.cv, self.n_jobs, self.logger
        )
        updated_model = hyperparam_searcher.search(model)

        # Update the model in the list
        self.models[model_index] = updated_model
        # Update best parameter list
        model_name = type(model.named_steps["model"]).__name__
        self.best_params[model_name] = hyperparam_searcher.best_params

        return updated_model

    def _get_best_model(self, metric="r2"):
        """Determine the best model using average cv score

        Parameters
        ----------
        metric : str
            Metric to use for determining the best model.

        Returns
        -------
        best_model : object
            Best model fitted on full data. If normalised, returns a pipeline with
            the scaler and the best model.
        """

        # best model name
        means = (
            self.scores_df.groupby(["model", "metric"])["score"]
            .mean()
            .unstack()
            .reset_index()
        )
        # get best model name
        best_model_name = means.loc[means[metric].idxmax(), "model"]

        # get best model with hyperparameters if available
        if len(self.best_params) > 0:
            best_model_params = self.best_params[best_model_name]
            print(best_model_params)
            best_model = MODEL_REGISTRY[best_model_name](**best_model_params)
        else:
            best_model = MODEL_REGISTRY[best_model_name]()

        if self.normalise:
            best_model = Pipeline([("scaler", self.scaler), ("model", best_model)])

        self.logger.info(
            f"{best_model_name} is the best model, refitting on full dataset..."
        )
        # refit best model on full dataset
        best_model.fit(self.X, self.y)

        return best_model

    def print_scores(self, model=None):
        # check if model is in self.models
        if model is not None:
            model_names = [type(model).__name__ for model in self.models]
            if model not in model_names:
                raise ValueError(
                    f"Model {model} not found. Available models are: {model_names}"
                )
        if model is None:
            means = (
                self.scores_df.groupby(["model", "metric"])["score"]
                .mean()
                .unstack()
                .reset_index()
            )
            print("Average scores across all models:")
            print(means)
        else:
            scores = self.scores_df[self.scores_df["model"] == model].pivot(
                index="fold", columns="metric", values="score"
            )
            print(f"Scores for {model} across all folds:")
            print(scores)

    def plot_results(self, model_name=None):
        """Plots the results of the cross-validation.

        Parameters
        ----------
        model_name : str
            Name of the model to plot. If None, plots best folds of each models.
            If a model name is specified, plots all folds of that model.
        """
        plot_results(self.cv_results, self.X, self.y, model_name=model_name)
