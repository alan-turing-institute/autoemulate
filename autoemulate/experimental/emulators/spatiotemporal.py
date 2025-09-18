from abc import abstractmethod

import torch
from autoemulate.core.types import OutputLike, TensorLike
from autoemulate.emulators.base import PyTorchBackend
from autoemulate.experimental.emulators.batch_prep import prepare_batch_fno
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
        loss_weighting: str = "uniform",
        loss_weights: list[float] | None = None,
        epochs: int = 100,
    ) -> None: # noqa: PLR0912, PLR0915
        """Train using autoregressive approach.

        Parameters
        ----------
        x : DataLoader
            Input data loader containing spatiotemporal sequences
        n_steps : int
            Number of autoregressive steps to train for
        training_mode : list[str] | None
            List of training features to activate. Options:
            - "rollout_prediction": Core autoregressive rollout
            - "teacher_forcing": Use true targets sometimes during training
            - "temporal_encoder": Use multiple input timesteps with encoding
            - "learnable_temporal_weights": Use learnable weights for historical steps
            - "multi_step": Predict multiple timesteps per forward pass
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
            ("uniform", "exponential", "final_only", "linear_decay", "learnable") # W291
        loss_weights : list[float] | None
            Custom loss weights for each timestep
        epochs : int
            Number of training epochs
        """
        # Setup components based on training mode
        device = (
            next(self.model.parameters()).device  # type: ignore
            if hasattr(self.model, "parameters")
            else "cpu"
        )

        if training_mode is None:
            training_mode = ["rollout_prediction"]

        # Store configuration for prediction step
        self._autoregressive_config = {
            "training_mode": training_mode,
            "t_in": t_in,
            "t_out": t_out,
            "step_size": step_size,
            "teacher_forcing_ratio": teacher_forcing_ratio,
        }

        # Regular temporal encoder
        if "temporal_encoder" in training_mode and t_in > 1:
            if self.temporal_encoder is None:
                channels = len(self.channels)
                self.temporal_encoder = torch.nn.Conv3d(
                    channels, channels, kernel_size=(t_in, 1, 1), padding=(0, 0, 0)
                )
            self.temporal_encoder = self.temporal_encoder.to(device)

        # Learnable temporal weights
        if "learnable_temporal_weights" in training_mode and t_in > 1:
            if self.temporal_weights is None:
                self.temporal_weights = torch.nn.Parameter(torch.ones(t_in) / t_in)
                channels = len(self.channels)
                self.temporal_projection = torch.nn.Conv2d(
                    channels, channels, kernel_size=1
                )
            self.temporal_weights = self.temporal_weights.to(device)
            self.temporal_projection = self.temporal_projection.to(device)

        # Multi-step prediction
        if "multi_step" in training_mode and step_size > 1:
            if self.output_projection is None:
                channels = len(self.channels)
                self.output_projection = torch.nn.Conv3d(
                    channels, channels * step_size, kernel_size=(1, 1, 1)
                )
            self.output_projection = self.output_projection.to(device)

        # Setup learnable loss weights
        if loss_weighting == "learnable":
            if self.loss_weights_params is not None:
                self.loss_weights_params = torch.nn.Parameter(torch.ones(n_steps))
            self.loss_weights_params = self.loss_weights_params.to(device)

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(x):
                current_input, targets = prepare_batch_fno(
                    batch, channels=self.channels, with_constants=False, with_time=True
                )
                # current_input: [B, C, T_in, H, W]
                # targets: [B, C, T_out, H, W]

                # Convert to [B, T_in, C, H, W] for processing
                current_input = current_input.permute(
                    0, 2, 1, 3, 4
                )  # [B, T_in, C, H, W]
                targets = targets.permute(0, 2, 1, 3, 4)  # [B, T_out, C, H, W]

                predictions = []
                total_loss = torch.tensor(0.0, device=current_input.device)

                # Calculate number of autoregressive steps
                n_ar_steps = (n_steps + step_size - 1) // step_size

                # Decide teacher forcing for this sequence
                use_teacher_forcing = (
                    "teacher_forcing" in training_mode
                    and torch.rand(1).item() < teacher_forcing_ratio
                )

                for step in range(n_ar_steps):
                    # Forward pass through model
                    if "learnable_temporal_weights" in training_mode and t_in > 1:
                        # Apply learnable temporal weights
                        batch_size, t_in_actual, channels, H, W = current_input.shape
                        weights = torch.softmax(self.temporal_weights, dim=0)

                        # Compute weighted sum: \sum_i \lambda_{-i} \cdot x_{t-i}
                        weighted_sum = torch.zeros(
                            batch_size, channels, H, W, device=current_input.device
                        )
                        for i in range(t_in_actual):
                            weighted_sum += weights[i] * current_input[:, i, :, :, :]

                        # Optional projection
                        if self.temporal_projection is not None:
                            model_input = self.temporal_projection(weighted_sum)
                        else:
                            model_input = weighted_sum

                    elif "temporal_encoder" in training_mode and t_in > 1:
                        # Apply temporal encoding
                        x_permuted = current_input.permute(
                            0, 2, 1, 3, 4
                        )  # [B, channels, t_in, H, W]
                        encoded = self.temporal_encoder(
                            x_permuted
                        )  # [B, channels, 1, H, W]
                        model_input = encoded.squeeze(2)  # [B, channels, H, W]
                    else:
                        # Use only the last timestep
                        model_input = current_input[:, -1]  # [B, channels, H, W]

                    # Get prediction
                    pred = self.forward(model_input)  # [B, channels, H, W]

                    # Handle multi-step prediction
                    if "multi_step" in training_mode and step_size > 1:
                        pred_expanded = pred.unsqueeze(2)  # [B, channels, 1, H, W]
                        multi_step = self.output_projection(
                            pred_expanded
                        )  # [B, channels*step_size, 1, H, W]
                        multi_step = multi_step.squeeze(
                            2
                        )  # [B, channels*step_size, H, W]

                        batch_size, _, H, W = multi_step.shape
                        channels = pred.shape[1]
                        multi_step = multi_step.view(
                            batch_size, channels, step_size, H, W
                        )
                        pred = multi_step.permute(
                            0, 2, 1, 3, 4
                        )  # [B, step_size, channels, H, W]
                    else:
                        pred = pred.unsqueeze(1)  # [B, 1, channels, H, W]

                    predictions.append(pred)

                    # Calculate loss for this step
                    start_idx = step * step_size
                    end_idx = min(start_idx + step_size, n_steps)
                    actual_steps = end_idx - start_idx

                    if start_idx < targets.shape[1]:
                        target = targets[:, start_idx:end_idx]
                        step_pred = pred[:, :actual_steps]

                        step_loss = self.loss_fn(step_pred, target)

                        # Apply loss weighting
                        if loss_weights is not None and step < len(loss_weights):
                            weight = loss_weights[step]
                        elif loss_weighting == "uniform":
                            weight = 1.0
                        elif loss_weighting == "exponential":
                            weight = torch.exp(torch.tensor(step * 0.1))
                        elif loss_weighting == "linear_decay":
                            weight = 1.0 - (step / n_ar_steps)
                        elif loss_weighting == "final_only":
                            weight = 1.0 if step == n_ar_steps - 1 else 0.1
                        elif loss_weighting == "learnable":
                            if hasattr(self, "loss_weights_params") and step < len(
                                self.loss_weights_params
                            ):
                                weights_normalized = torch.softmax(
                                    self.loss_weights_params, dim=0
                                )
                                weight = weights_normalized[step]
                            else:
                                weight = 1.0
                        else:
                            weight = 1.0

                        weighted_loss = step_loss * weight
                        total_loss += weighted_loss

                    # Prepare input for next step
                    if step < n_ar_steps - 1:
                        if use_teacher_forcing:
                            # Use ground truth
                            next_input = targets[
                                :, start_idx : start_idx + actual_steps
                            ]
                        else:
                            # Use prediction
                            next_input = pred[:, :actual_steps]

                        # Update sliding window input
                        current_input = torch.cat(
                            [
                                current_input[:, actual_steps:],  # Remove old timesteps
                                next_input,  # Add new timesteps
                            ],
                            dim=1,
                        )

                epoch_loss += total_loss.item()

                # Optimization step
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.5e}"  # noqa: E501
                    )

            print(f"Epoch {epoch} completed. Average Loss: {epoch_loss / len(x):.5e}")

            # Optional: Print learned weights
            if "learnable_temporal_weights" in training_mode and epoch % 20 == 0:
                weights = torch.softmax(self.temporal_weights, dim=0)
                print(f"Temporal weights: {weights.detach().cpu().numpy()}")

            if loss_weighting == "learnable" and epoch % 20 == 0:
                weights = torch.softmax(self.loss_weights_params, dim=0)
                print(f"Loss weights: {weights.detach().cpu().numpy()}")

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
                # Use prepare_batch_fno to get properly formatted input
                x, _ = prepare_batch_fno(
                    current_sample,
                    channels=self.channels,
                    with_constants=False,
                    with_time=True,
                )
                # Forward pass
                pred = self(x)  # [batch, channels, time, height, width]
                print(pred.shape)
                assert isinstance(pred, TensorLike)
                predictions.append(pred)
                # Update input for next iteration
                # Convert prediction back to input format
                current_input = pred.permute(0, 2, 3, 4, 1)
        # Stack all predictions along time dimension
        return torch.cat(predictions, dim=2)
