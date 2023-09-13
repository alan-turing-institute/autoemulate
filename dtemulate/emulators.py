from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
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

        :param X: Input data (simulation input).
        :param y: Target data (simulation output).
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predicts the output of the simulator for a given input."""
        pass

    @abstractmethod
    def score(self, X, y):
        """Returns the score of the emulator."""
        pass


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

        :param X: Input data (simulation input).
        :param y: Target data (simulation output).
        """
        self.model = mogp_emulator.GaussianProcess(X, y, *self.args, **self.kwargs)
        self.model = mogp_emulator.fit_GP_MAP(self.model)

    def predict(self, X):
        """Predicts the output of the simulator for a given input.

        :param X: Input data (simulation input).
        """
        if self.model is not None:
            return self.model.predict(X)
        else:
            raise ValueError("Emulator not fitted yet.")

    def score(self, X, y):
        """Returns the score of the emulator."""
        predictions_means = self.predict(X).mean
        rmse = np.sqrt(mean_squared_error(y, predictions_means)).round(2)
        return rmse


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

        :param X: Input data (simulation input).
        :param y: Target data (simulation output).
        """
        self.model.fit(X, y)

    def predict(self, X):
        """Predicts the output of the simulator for a given input.

        :param X: Input data (simulation input).
        """
        return self.model.predict(X)

    def score(self, X, y):
        """Returns the score of the emulator."""
        predictions = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions)).round(2)
        return rmse
