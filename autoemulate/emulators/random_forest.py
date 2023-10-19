import numpy as np

from sklearn.ensemble import RandomForestRegressor
from autoemulate.emulators import Emulator


class RandomForest(Emulator):
    """Random forest Emulator.

    Implements Random Forests regression from scikit-learn.
    """

    def __init__(self, n_estimators=100, *args, **kwargs):
        """Initializes a RandomForest object."""
        self.args = args
        self.kwargs = {"n_estimators": n_estimators, **kwargs}
        self.model = RandomForestRegressor(*self.args, **self.kwargs)

    def fit(self, X, y):
        """Fits the emulator to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """Predicts the output of the simulator for a given input.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.

        Returns
        -------
        predictions : numpy.ndarray
            Predictions of the emulator.
        """
        return self.model.predict(X)

    def score(self, X, y, metric):
        """Returns the score of the emulator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.
        metric : str
            Name of the metric to use, currently either rsme or r2.
        Returns
        -------
        metric : float
            Metric of the emulator.

        """
        predictions = self.predict(X)
        return metric(y, predictions)
