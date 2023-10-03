from sklearn.model_selection import KFold
from autoemulate.experimental_design import LatinHypercube
from autoemulate.metrics import METRIC_REGISTRY
from autoemulate.emulators import MODEL_REGISTRY
from autoemulate.cv import CV_REGISTRY
import pandas as pd
import numpy as np


class AutoEmulate:
    """Automatically compares emulators."""

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

    def setup(self, X, y, fold_strategy="kfold", folds=5):
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
        self._check_data(X, y)
        self._preprocess_data(X, y)
        self.models = [
            MODEL_REGISTRY[model_name]() for model_name in MODEL_REGISTRY.keys()
        ]
        self.metrics = [metric_name for metric_name in METRIC_REGISTRY.keys()]
        self.cv = CV_REGISTRY[fold_strategy](folds=folds, shuffle=True)
        self.is_set_up = True

    def compare(self):
        """Compares the emulators."""
        if not self.is_set_up:
            raise RuntimeError("Must run setup() before compare()")

        print(f"Starting {self.cv}-fold cross-validation...")

        for model in self.models:
            model_name = type(model).__name__
            print(f"Training {model_name}...")
            self._score_model_with_cv(model)

    def print_scores(self, model=None):
        """Prints the scores of the emulators.

        Parameters
        ----------
        model : str, optional
            If model is None, prints the average scores across all models.
            Otherwise, prints the scores for the specified model across folds.

        """
        if model is None:
            means = (
                self.scores_df.groupby(["model", "metric"])["score"]
                .mean()
                .unstack()
                .reset_index()
                .sort_values(by="r2", ascending=False)
            )
            print("Average scores across all models:")
            print(means.to_string(index=False))
           
        else:
            specific_model_scores = self.scores_df[self.scores_df["model"] == model]
            folds = (
                specific_model_scores.groupby(["metric", "fold"])["score"]
                .mean()
                .unstack()
                .transpose()
            )
            folds.columns.name = None
            folds.index.name = None
            folds.loc["Mean"] = folds.mean()
            folds.loc["Std Dev"] = folds.std()
            print(f"Scores for {model} across all folds:")
            print(folds.to_string())
            

    def _check_data(self, X, y):
        """Validates data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("X and y should not contain NaNs.")
        
        
    def _preprocess_data(self, X, y):
        """Preprocesses data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.

        """
        self.X = np.array(X)
        self.y = np.array(y)


    def _train_model(self, model, X, y):
        """Trains the model.

        Parameters
        ----------
        model : object
            The model to train.
        X : array-like, shape (n_samples, n_features)
            Simulation input.

        Returns
        -------
        model : object
            The trained model.
        """
        model.fit(X, y)
        return model

    def _evaluate_model(self, trained_model, X, y):
        """Evaluates the model.

        Parameters
        ----------
        trained_model : object
            The trained model.
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.

        Returns
        -------
        scores : dict
            The scores of the model.
        """
        scores = {}
        for metric in self.metrics:
            metric_func = METRIC_REGISTRY[metric]
            score = trained_model.score(X, y, metric=metric_func)
            scores[metric] = score
        return scores

    def _score_model_with_cv(self, model):
        """Scores the model using cross-validation.

        Parameters
        ----------
        model : object
            The model to score.

        Returns
        -------
        scores_df : pandas.DataFrame
            The scores of the model.
        """
        cv = self.cv
        model_name = type(model).__name__

        for fold, (train_index, test_index) in enumerate(cv.split(self.X)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            trained_model = self._train_model(model, X_train, y_train)
            fold_scores = self._evaluate_model(trained_model, X_test, y_test)

            for metric, score in fold_scores.items():
                new_row = pd.DataFrame(
                    {
                        "model": [model_name],
                        "metric": [metric],
                        "fold": [fold],  # Now correctly included
                        "score": [score],
                    }
                )
                self.scores_df = pd.concat([self.scores_df, new_row], ignore_index=True)
