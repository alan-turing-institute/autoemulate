from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mogp_emulator
import numpy as np


class Emulator(ABC):
    """An abstract base class for emulators."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initializes an Emulator object."""
        pass

    @abstractmethod
    def fit(self, X, y):
        """Fits the emulator to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.

        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predicts the output of the simulator for a given input.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        """
        pass

    @abstractmethod
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
        """
        pass


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

        Returns
        -------
        metric : float
            Metric of the emulator.

        """
        predictions = self.predict(X)
        return metric(y, predictions)


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


class GaussianProcess(Emulator):
    """Gaussian process Emulator.

    Implements GaussianProcess regression from the mogp_emulator package.
    """

    def __init__(self, nugget="fit", *args, **kwargs):
        """Initializes a GaussianProcess object."""
        self.args = args
        self.kwargs = {"nugget": nugget, **kwargs}
        self.model = None

    def fit(self, X, y):
        """Fits the emulator to the data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data (simulation input).
        y : numpy.ndarray
            Target data (simulation output).
        """
        self.model = mogp_emulator.GaussianProcess(X, y, *self.args, **self.kwargs)
        self.model = mogp_emulator.fit_GP_MAP(self.model)

    def predict(self, X):
        """Predicts the output of the simulator for a given input.

        Parameters
        ----------
        X : numpy.ndarray
            Input data (simulation input).

        Returns
        -------
        predictions : numpy.ndarray
            Predictions of the emulator.
        """
        if self.model is not None:
            return self.model.predict(X).mean
        else:
            raise ValueError("Emulator not fitted yet.")

    def score(self, X, y, metric):
        """Returns the score of the emulator.

        Parameters
        ----------
        X : numpy.ndarray
            Input data (simulation input).
        y : numpy.ndarray
            Target data (simulation output).

        Returns
        -------
        rmse : float
            Root mean squared error of the emulator.
        """
        prediction_means = self.predict(X)
        return metric(y, prediction_means)


MODEL_REGISTRY = {
    "GaussianProcess": GaussianProcess,
    "RandomForest": RandomForest,
    "GaussianProcess2": GaussianProcess2,
    "NeuralNetwork": NeuralNetwork,
}
