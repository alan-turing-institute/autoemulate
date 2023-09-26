from sklearn.model_selection import KFold
from autoemulate.experimental_design import LatinHypercube
from autoemulate.emulators import (
    GaussianProcess,
    RandomForest,
    GaussianProcess2,
    NeuralNetwork,
)
from autoemulate.metrics import METRIC_REGISTRY
import numpy as np
import pandas as pd


class AutoEmulate:
    """Automatically compares emulators."""

    def __init__(self):
        """Initializes an AutoEmulate object."""
        self.X = None
        self.y = None
        self.cv = None
        self.models = None
        self.scores = {}
        self.fitted_models = {}

    def compare(self, X, y, cv=5, models=None, metrics=None):
        """ "Compares emulators using cross-validation.

        Parameters
        ----------
        X : numpy.ndarray
            Input data (simulation input).
        y : numpy.ndarray
            Target data (simulation output).
        cv : int
            Number of folds for cross-validation.
        models : list
            List of emulators to compare.

        Returns
        -------
        scores : dict
            Dictionary of scores for each emulator.
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.cv = cv
        self.models = (
            models
            if models
            else [
                GaussianProcess(),
                RandomForest(),
                GaussianProcess2(),
                NeuralNetwork(),
            ]
        )

        if metrics is None:
            metrics = METRIC_REGISTRY.keys()

        self.scores = {
            type(model).__name__: {metric: 0 for metric in metrics}
            for model in self.models
        }

        # Validation checks, same as before
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if np.isnan(self.X).any() or np.isnan(self.y).any():
            raise ValueError("X and y should not contain NaNs.")

        print(f"Starting {self.cv}-fold cross-validation...")

        for model in self.models:
            model_name = type(model).__name__
            print(f"Training {model_name}...")
            metric_fold_scores = {metric: [] for metric in metrics}

            kfold = KFold(n_splits=self.cv, shuffle=True)
            for train_index, test_index in kfold.split(self.X):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                model.fit(X_train, y_train)

                for metric in metrics:
                    metric_func = METRIC_REGISTRY[metric]
                    score = model.score(X_test, y_test, metric=metric_func)
                    metric_fold_scores[metric].append(score)

            for metric, scores in metric_fold_scores.items():
                self.scores[model_name][metric] = np.mean(scores)

    def print_scores(self):
        """Prints scores for each emulator in a table format."""
        # Convert the nested dictionary to a Pandas DataFrame
        df_scores = pd.DataFrame(self.scores).T

        print(f"Scores (average over {self.cv} folds):")
        print(df_scores.to_string())

    def get_fitted_models(self):
        """Returns the fitted models."""
        return self.fitted_models
