import numpy as np
import mogp_emulator
from autoemulate.emulators import Emulator


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

    def predict(self, X, return_std=False):
        """Predicts the output of the simulator for a given input.

        Parameters
        ----------
        X : numpy.ndarray
            Input data (simulation input).
        return_var : bool
            If True, returns a touple with two ndarrays,
            one with the mean and one with the standard deviations of the prediction.

        Returns
        -------
        predictions : numpy.ndarray
            Predictions of the emulator.
        """
        if return_std:
            mean, var, _ = self.model.predict(X)
            return mean, np.sqrt(var)
        else:
            return self.model.predict(X).mean

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
