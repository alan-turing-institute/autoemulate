from autoemulate.metrics import METRIC_REGISTRY
from autoemulate.emulators import MODEL_REGISTRY
from autoemulate.cv import CV_REGISTRY
from autoemulate.logging_config import configure_logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# impoprt check_X_y from sklearn
from sklearn.utils.validation import check_X_y
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import logging

verbose = 0  # or 1, based on user input or command-line argument
configure_logging(verbose)


class AutoEmulate:
    def __init__(self):
        """Initializes an AutoEmulate object."""
        self.X = None
        self.y = None
        self.scores_df = pd.DataFrame(
            columns=["model", "metric", "fold", "score"]
        ).astype(
            {"model": "object", "metric": "object", "fold": "int64", "score": "float64"}
        )
        self.is_set_up = False

    def setup(self, X, y, hyperparameter_search=False, fold_strategy="kfold", folds=5):
        """Sets up the AutoEmulate object.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.
        fold_strategy : str
            Cross-validation strategy, currently either "kfold" or "stratified_kfold".
        folds : int
            Number of folds.

        """
        self.X, self.y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self.models = [model() for model in MODEL_REGISTRY.values()]
        self.metrics = [metric for metric in METRIC_REGISTRY.keys()]
        self.cv = CV_REGISTRY[fold_strategy](folds=folds, shuffle=True)
        self.is_set_up = True
        self.hyperparameter_search = hyperparameter_search
        self.logger = logging.getLogger(type(self).__name__)

    def compare(self):
        if not self.is_set_up:
            raise RuntimeError("Must run setup() before compare()")

        for model in self.models:
            model_name = type(model).__name__

            if self.hyperparameter_search:
                self.perform_hyperparameter_search_for_model(model)

            self.logger.info(f"Training {model_name}...")
            self.logger.info(f"Parameters: {model.get_params()}")
            for metric_name in self.metrics:
                scorer = make_scorer(METRIC_REGISTRY[metric_name])
                scores = cross_val_score(
                    model, self.X, self.y, cv=self.cv, scoring=scorer
                )
                for fold, score in enumerate(scores):
                    new_row = pd.DataFrame(
                        {
                            "model": [model_name],
                            "metric": [metric_name],
                            "fold": [fold],
                            "score": [score],
                        }
                    )
                    self.scores_df = pd.concat(
                        [self.scores_df, new_row], ignore_index=True
                    )

    def perform_hyperparameter_search_for_model(self, model):
        model_name = type(model).__name__
        self.logger.info(f"Performing grid search for {model_name}...")
        param_grid = (
            model.get_grid_params()
        )  # Assumes that each model has a `get_grid_params` method
        grid_search = GridSearchCV(model, param_grid, cv=self.cv)
        grid_search.fit(self.X, self.y)
        best_params = grid_search.best_params_
        self.logger.info(f"Best parameters for {model_name}: {best_params}")
        model.set_params(**best_params)  # Update the model with the best parameters

    def print_scores(self, model=None):
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
            scores = (
                self.scores_df[self.scores_df["model"] == model]
                .pivot(index="fold", columns="metric", values="score")
                .pipe(print)
            )
            print(f"Scores for {model} across all folds:")
            print(scores)
