from abc import abstractmethod
from typing import Literal

import torch
from autoemulate.core.types import OutputLike, TensorLike
from autoemulate.emulators.base import PyTorchBackend
from torch import nn
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

    def fit_autoregressive(
        self,
        x: DataLoader,
        n_steps: int,
        training_mode: list[str] | None = None,
        t_in: int = 5,
        t_out: int = 10,
        step_size: int = 1,
        teacher_forcing_ratio: float = 0.5,
        loss_weighting: list[str] | None = None,
        loss_weights: list[float] | None = None,
        epochs: int = 100,
        **kwargs,
    ) -> None:
        """Train using autoregressive approach.

        Parameters
        ----------
        x : DataLoader
            Input data loader containing spatiotemporal sequences
        n_steps : int
            Number of autoregressive steps to train for
        training_mode : List[str]
            List of training features to activate. Options:
            - "rollout_prediction": Core autoregressive rollout
            - "teacher_forcing": Use true targets sometimes during training
            - "temporal_encoder": Use multiple input timesteps with encoding
            - "learnable_temporal_weights": Use learnable weights for
            historical timesteps
            - "multi_step": Predict multiple timesteps per forward pass
            - "loss_weighting": Apply weighted loss across timesteps
        t_in : int
            Number of input timesteps (for temporal_encoder mode)
        t_out : int
            Number of output timesteps to predict
        step_size : int
            Number of timesteps to predict per forward pass (for multi_step mode)
        teacher_forcing_ratio : float
            Ratio of teacher forcing (0.0 = never, 1.0 = always)
        loss_weighting : str
            How to weight losses across timesteps
        loss_weights : Optional[List[float]]
            Custom loss weights for each timestep
        epochs : int
            Number of training epochs
        """
        # Store configuration for prediction consistency
        self._autoregressive_config = {
            "training_mode": training_mode,
            "t_in": t_in,
            "t_out": t_out,
            "step_size": step_size,
            "teacher_forcing_ratio": teacher_forcing_ratio,
        }

        if "temporal_encoder" in training_mode and t_in > 1:
            if self.temporal_encoder is None:
                n_var = 1
                self.temporal_encoder = torch.nn.Conv3d(
                    n_var, n_var, kernel_size=(t_in, 1, 1), padding=(0, 0, 0)
                )
                self.temporal_encoder = self.temporal_encoder.to(self.device)

            else:
                self.temporal_encoder = self.temporal_encoder.to(self.device)

        # Learnable temporal weights
        if "learnable_temporal_weights" in training_mode and t_in > 1:
            if self.temporal_weights is None:
                # Initialise learnable weights lamda_i for each historical timestep
                self.temporal_weights = nn.Parameter(
                    torch.ones(t_in) / t_in
                )  # one weight per input timestep

                # We could potentially add a projection layer after weighted combination
                # this needs to be benchmarked
                n_vars = 1
                self.temporal_projection = nn.Conv2d(n_vars, n_vars, kernel_size=1)

            self.temporal_weights = self.temporal_weights.to(self.device)
            if self.temporal_projection is not None:
                self.temporal_projection = self.temporal_projection.to(self.device)

        # Setup learnable loss weights
        if loss_weighting == "learnable":
            if self.loss_weights_params is None:
                self.loss_weights_params = nn.Parameter(torch.ones(n_steps))
            self.loss_weights_params = self.loss_weights_params.to(self.device)

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
