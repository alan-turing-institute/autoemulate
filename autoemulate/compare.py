from sklearn.model_selection import KFold
from autoemulate.experimental_design import LatinHypercube

# from autoemulate.emulators import (
#     GaussianProcess,
#     RandomForest,
#     GaussianProcess2,
#     NeuralNetwork,
# )
from autoemulate.metrics import METRIC_REGISTRY
from autoemulate.emulators import MODEL_REGISTRY
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
        self.is_set_up = False

    def setup(self, X, y, cv=None):
        self._preprocess_data(X, y)
        self.cv = cv if cv else 5
        self.models = [
            MODEL_REGISTRY[model_name]() for model_name in MODEL_REGISTRY.keys()
        ]
        self.metrics = [metric_name for metric_name in METRIC_REGISTRY.keys()]
        self.is_set_up = True

    def compare(self):
        if not self.is_set_up:
            raise RuntimeError("Must run setup() before compare()")

        print(f"Starting {self.cv}-fold cross-validation...")

        for model in self.models:
            model_name = type(model).__name__
            print(f"Training {model_name}...")
            metric_fold_scores = self._score_model_with_cv(model)

            self.scores[model_name] = {
                metric: {
                    "mean": np.mean(scores),
                    "all_folds": scores,
                }
                for metric, scores in metric_fold_scores.items()
            }

    def print_scores(self, model=None):
        pd.set_option("display.float_format", lambda x: "%.3f" % x)

        if model is None:
            # Create a DataFrame from self.scores but only take the 'mean' values
            df_means = pd.DataFrame(
                {
                    model: {
                        metric: details["mean"] for metric, details in metrics.items()
                    }
                    for model, metrics in self.scores.items()
                }
            ).T

            print("Average Scores Across All Models:")
            print(df_means.to_string())
        else:
            # Extract the scores for the specified model
            model_scores = self.scores.get(model, {})

            # Create a DataFrame from the 'all_folds' scores
            df_folds = pd.DataFrame(
                {
                    metric: details["all_folds"]
                    for metric, details in model_scores.items()
                }
            )

            # Add mean and standard deviation rows at the end
            df_folds.loc["Mean"] = df_folds.mean()
            df_folds.loc["Std Dev"] = df_folds.std()

            print(f"Scores for {model} Across All Folds:")
            print(df_folds.to_string())

    def _preprocess_data(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if np.isnan(self.X).any() or np.isnan(self.y).any():
            raise ValueError("X and y should not contain NaNs.")

    def _train_model(self, model, X, y):
        model.fit(X, y)
        return model

    def _evaluate_model(self, trained_model, X, y):
        scores = {}
        for metric in self.metrics:
            metric_func = METRIC_REGISTRY[metric]
            score = trained_model.score(X, y, metric=metric_func)
            scores[metric] = score
        return scores

    def _score_model_with_cv(self, model):
        metric_fold_scores = {metric: [] for metric in self.metrics}
        kfold = KFold(n_splits=self.cv, shuffle=True)

        for train_index, test_index in kfold.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            trained_model = self._train_model(model, X_train, y_train)
            fold_scores = self._evaluate_model(trained_model, X_test, y_test)

            for metric, score in fold_scores.items():
                metric_fold_scores[metric].append(score)

        return metric_fold_scores
