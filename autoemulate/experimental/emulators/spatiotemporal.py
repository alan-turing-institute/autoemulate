from abc import abstractmethod

from autoemulate.core.types import OutputLike, TensorLike
from autoemulate.emulators.base import PyTorchBackend
from torch.utils.data import DataLoader


class SpatioTemporalBackend(PyTorchBackend):
    """A spatio-temporal backend for emulators."""

    def fit(self, x: TensorLike | DataLoader, y: TensorLike | None = None):
        """Train a spatio-temporal emulator.

        Parameters
        ----------
        x: TensorLike | DataLoader
            Input features as `TensorLike` or `DataLoader`.
        y: OutputLike | None
            Target values (not needed if x is a DataLoader).

        """
        if isinstance(x, TensorLike) and isinstance(y, TensorLike):
            return super().fit(x, y)
        if isinstance(x, DataLoader) and y is None:
            return self._fit(x, y)
        msg = "Invalid input types. Expected pair of TensorLike or DataLoader only."
        raise RuntimeError(msg)

    @abstractmethod
    def _fit(self, x: TensorLike | DataLoader, y: TensorLike | None = None): ...

    def predict(
        self, x: TensorLike | DataLoader, with_grad: bool = False
    ) -> OutputLike:
        """Predict the output for the given input.

        Parameters
        ----------
        x: TensorLike | DataLoader
            Input `TensorLike` to make predictions for or `DataLoader`.
        with_grad: bool
            Whether to enable gradient calculation. Defaults to False.

        Returns
        -------
        OutputLike
            The emulator predicted spatiotemporal output.
        """
        if isinstance(x, TensorLike):
            return super().predict(x, with_grad)
        return self._predict(x, with_grad)

    @abstractmethod
    def _predict(self, x: TensorLike | DataLoader, with_grad: bool) -> OutputLike: ...
