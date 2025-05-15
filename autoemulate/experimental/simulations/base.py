from abc import ABC, abstractmethod

from autoemulate.experimental.data.validation import ValidationMixin
from autoemulate.experimental.types import TensorLike


class Simulator(ABC, ValidationMixin):
    """
    Base class for simulations. All simulators should inherit from this class.
    This class provides the interface and common functionality for different
    simulation implementations.
    """

    @abstractmethod
    def sample_inputs(self, n: int) -> TensorLike:
        """
        Abstract method to generate random input samples.

        Parameters
        ----------
        n : int
            Number of input samples to generate.

        Returns
        -------
        TensorLike
            Random input tensor.
        """

    @abstractmethod
    def _forward(self, x: TensorLike) -> TensorLike:
        """
        Abstract method to perform the forward simulation.

        Parameters
        ----------
        x : TensorLike
            Input tensor.

        Returns
        -------
        TensorLike
            Simulated output tensor.
        """

    def forward(self, x: TensorLike) -> TensorLike:
        """
        Generate samples from input data using the simulator. Combines the
        abstract method `_forward` with some validation checks.

        Parameters
        ----------
        x : TensorLike
            Input tensor of shape (n_samples, n_features).

        Returns
        -------
        TensorLike
            Simulated output tensor.
        """
        y = self.check_matrix(self._forward(self.check_matrix(x)))
        x, y = self.check_pair(x, y)
        return y
