from sklearn.model_selection import KFold
from autoemulate.experimental_design import LatinHypercube
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
        self.scores_df = pd.DataFrame(columns=["model", "metric", "fold", "score"]).astype({'model': 'object', 'metric': 'object', 'fold': 'int64', 'score': 'float64'})
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
            self._score_model_with_cv(model)
            

    def print_scores(self, model=None):
        if model is None:
            means = self.scores_df.groupby(["model", "metric"])["score"].mean().unstack()
            print("Average Scores Across All Models:")
            print(means)
        else:
            specific_model_scores = self.scores_df[self.scores_df["model"] == model]
            folds = specific_model_scores.groupby(["metric", "fold"])["score"].mean().unstack()
            folds.loc["Mean"] = folds.mean()
            folds.loc["Std Dev"] = folds.std()
            print(f"Scores for {model} Across All Folds:")
            print(folds)

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
        kfold = KFold(n_splits=self.cv, shuffle=True)
        model_name = type(model).__name__

        for fold, (train_index, test_index) in enumerate(kfold.split(self.X)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            trained_model = self._train_model(model, X_train, y_train)
            fold_scores = self._evaluate_model(trained_model, X_test, y_test)

            for metric, score in fold_scores.items():
                new_row = pd.DataFrame({
                    "model": [model_name],
                    "metric": [metric],
                    "fold": [fold],  # Now correctly included
                    "score": [score]
                })
                self.scores_df = pd.concat([self.scores_df, new_row], ignore_index=True)







