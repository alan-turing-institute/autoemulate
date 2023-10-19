from abc import ABC, abstractmethod


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
    def predict(self, X, return_std=False):
        """Predicts the output of the simulator for a given input.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        return_std : bool
            If True, returns a touple with two ndarrays,
            one with the mean and one with the standard deviations of the prediction.
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
