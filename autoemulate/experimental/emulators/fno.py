import inspect

import torch
from autoemulate.core.types import OutputLike, TensorLike
from autoemulate.experimental.emulators.spatiotemporal import SpatioTemporalEmulator
from neuralop.models import FNO
from the_well.benchmark.metrics import VRMSE
from the_well.data.datasets import WellMetadata
from torch import nn
from torch.utils.data import DataLoader


def prepare_batch(sample, channels=(0,), with_constants=True, with_time=False):
    """Prepare a batch of input and output data."""
    # Get input fields, constant scalars and output fields
    x = sample["input_fields"][
        :, :, :, :, channels
    ]  # [batch, time, height, width, len(channels)]
    constant_scalars = sample["constant_scalars"]  # [batch, n_constants]
    y = sample["output_fields"][
        :, :, :, :, channels
    ]  # [batch, time, height, width, len(channels)]

    batch_size = x.shape[0]

    # Permute both x and y
    x = x.permute(0, 4, 1, 2, 3)  # [batch, len(channels), time, height, width]
    y = y.permute(0, 4, 1, 2, 3)  # [batch, len(channels), time, height, width]

    # Only add constants to input, not output
    if with_constants:
        # Assign spatio-temporal dims to constants
        time_window, height, width = x.shape[2], x.shape[3], x.shape[4]
        n_constants = constant_scalars.shape[-1]

        # Add spatio-temporal dims to constants
        c_broadcast = constant_scalars.reshape(batch_size, n_constants, 1, 1, 1).expand(
            batch_size, n_constants, time_window, height, width
        )

        # Concatenate along channel dimension
        x = torch.cat([x, c_broadcast], dim=1)

    if not with_time:
        # Take last time step for both input and output
        return x[:, :, -1, :, :], y[:, :, -1, :, :]
    # Otherwise include time
    return x, y


class FNOEmulator(SpatioTemporalEmulator):
    """An FNO emulator."""

    def __init__(
        self,
        x=None,
        y=None,
        channels: tuple[int, ...] = (0,),
        metadata: WellMetadata | None = None,
        with_constants: bool = False,
        with_time: bool = False,
        n_steps_output: int = 1,
        loss_fn: nn.Module | None = None,
        **kwargs,
    ):
        _, _ = x, y  # Unused
        # Ensure parent initialisers run before creating nn.Module attributes
        super().__init__()
        self.model = FNO(**kwargs)
        self.channels = channels
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.n_epochs = 10
        self.metadata = metadata
        self.loss_fn = loss_fn or VRMSE()
        self.with_time = with_time
        self.with_constants = with_constants
        self.n_steps_output = n_steps_output
        if (
            "metadata" in inspect.signature(self.model.__call__).parameters
            and metadata is None
        ):
            msg = "metadata must be provided if model requires it"
            raise ValueError(msg)

    @staticmethod
    def is_multioutput() -> bool:  # noqa: D102
        return True

    def _fit(
        self, x: TensorLike | DataLoader | None = None, y: TensorLike | None = None
    ):
        assert isinstance(x, DataLoader), "x currently must be a DataLoader"
        assert y is None, "y currently must be None"

        for epoch in range(self.n_epochs):
            for idx, batch in enumerate(x):
                # print(batch["input_fields"].shape)
                # Prepare input with constants
                # print(batch["input_fields"].shape, batch["output_fields"].shape)
                x_batch, y_batch = prepare_batch(
                    batch,
                    channels=self.channels,
                    with_constants=self.with_constants,
                    with_time=self.with_time,
                )

                # Predictions
                y_pred = self.model(x_batch)

                if self.with_time:
                    y_pred = y_pred[:, :, self.n_steps_output, ...]  # B, C, T, ...

                # Get loss
                loss = self.loss_fn(y_pred, y_batch, self.metadata).mean()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                print(f"epoch {epoch}, sample {idx:5d}, loss: {loss.item():.5e}")

    def forward(self, x: TensorLike):
        """Forward pass."""
        return self.model(x)

    def _predict(self, x: TensorLike | DataLoader, with_grad: bool) -> OutputLike:
        assert isinstance(x, DataLoader), "x currently must be a DataLoader"
        with torch.set_grad_enabled(with_grad):
            channels = (0,)  # Which channel to use
            all_preds = []
            for _, batch in enumerate(x):
                # Prepare input with constants
                x, _ = prepare_batch(
                    batch,
                    channels=channels,
                    with_constants=self.with_constants,
                    with_time=self.with_time,
                )
                out = self(x)
                if self.with_time:
                    out = out[:, :, : self.n_steps_output, ...]  # B, C, T, ...
                all_preds.append(out)
            return torch.cat(all_preds)
