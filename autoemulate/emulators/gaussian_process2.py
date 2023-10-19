import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from autoemulate.emulators import Emulator


class GaussianProcess2(Emulator):
    """Gaussian process Emulator.

    Implements GaussianProcessRegressor from scikit-learn.
    """

    def __init__(self, *args, **kwargs):
        """Initializes a GaussianProcess object."""
        self.args = args
        self.kwargs = kwargs
        self.model = GaussianProcessRegressor(*self.args, **self.kwargs)

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

    def predict(self, X, return_std=False):
        """Predicts the output of the simulator for a given input.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        return_std : bool
            If True, returns a touple with two ndarrays,
            one with the mean and one with the variance of the prediction.

        Returns
        -------
        predictions : numpy.ndarray
            Predictions of the emulator.
        """
        return self.model.predict(X, return_std=return_std)

    def score(self, X, y, metric):
        """Returns the score of the emulator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.

        Returns
        -------
        metric : float
            Metric of the emulator.

        """
        predictions = self.predict(X)
        return metric(y, predictions)
