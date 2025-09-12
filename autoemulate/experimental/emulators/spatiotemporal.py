from abc import abstractmethod

import torch
from autoemulate.core.types import OutputLike, TensorLike
from autoemulate.emulators.base import PyTorchBackend
from torch.utils.data import DataLoader


class SpatioTemporalEmulator(PyTorchBackend):
    """A spatio-temporal backend for emulators."""

    channels: tuple[int, ...]

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
        self,
        x: TensorLike | DataLoader,
        with_grad: bool = False,
    ) -> OutputLike:
        """Predict the output for the given input.

        Parameters
        ----------
        x: TensorLike | DataLoader
            Input `TensorLike` or `DataLoader` to make predictions for.
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

    # TODO: update to handle DataLoader input as part of #814 and #815
    def predict_autoregressive(self, initial_sample: dict, n_steps: int) -> TensorLike:
        """Perform autoregressive prediction."""
        from autoemulate.experimental.emulators.fno import (  # noqa: PLC0415 to avoid circular import
            prepare_batch,
        )

        self.eval()
        predictions = []
        # Get initial input
        current_input = initial_sample[
            "input_fields"
        ]  # [batch, time, height, width, channels]
        constant_scalars = initial_sample["constant_scalars"]
        with torch.no_grad():
            for _ in range(n_steps):
                # Prepare current input in the format expected by prepare_batch
                current_sample = {
                    "input_fields": current_input,
                    "output_fields": current_input,  # dummy, not used
                    "constant_scalars": constant_scalars,
                }
                # Use prepare_batch to get properly formatted input
                x, _ = prepare_batch(
                    current_sample,
                    channels=self.channels,
                    with_constants=True,
                    with_time=True,
                )
                # Forward pass
                pred = self(x)  # [batch, channels, time, height, width]
                assert isinstance(pred, TensorLike)
                predictions.append(pred)
                # Update input for next iteration
                # Convert prediction back to input format
                current_input = pred.permute(0, 2, 3, 4, 1)
        # Stack all predictions along time dimension
        return torch.cat(predictions, dim=2)
