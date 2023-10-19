import numpy as np

from sklearn.neural_network import MLPRegressor
from autoemulate.emulators import Emulator


class NeuralNetwork(Emulator):
    """Multi-layer perceptron Emulator.

    Implements MLPRegressor from scikit-learn.
    """

    def __init__(self, *args, **kwargs):
        """Initializes an MLPRegressor object."""
        self.args = args
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y):
        """Fits the emulator to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.
        """
        # just to test, for now
        hidden_layers = (X.shape[1] * 2, X.shape[1])
        self.model = MLPRegressor(
            solver="adam",
            alpha=1e-5,
            hidden_layer_sizes=hidden_layers,
            max_iter=1000,
            *self.args,
            **self.kwargs
        )
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

        if self.model is not None:
            return self.model.predict(X)
        else:
            raise ValueError("Emulator not fitted yet.")

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
