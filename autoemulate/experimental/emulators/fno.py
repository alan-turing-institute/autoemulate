import torch
from autoemulate.core.types import OutputLike, TensorLike
from autoemulate.experimental.emulators.batch_prep import prepare_batch_fno
from autoemulate.experimental.emulators.spatiotemporal import SpatioTemporalEmulator
from neuralop.models import FNO
from torch.utils.data import DataLoader


class FNOEmulator(SpatioTemporalEmulator):
    """An FNO emulator."""

    def __init__(
        self, x=None, y=None, channels: tuple[int, ...] = (0,), *args, **kwargs
    ):
        _, _ = x, y  # Unused
        # Ensure parent initialisers run before creating nn.Module attributes
        super().__init__()
        self.model = FNO(**kwargs)
        self.channels = channels
        self.optimizer = torch.optim.Adam(self.model.parameters())

    @staticmethod
    def is_multioutput() -> bool:  # noqa: D102
        return True

    def _fit(self, x: TensorLike | DataLoader, y: TensorLike | None = None):
        assert isinstance(x, DataLoader), "x currently must be a DataLoader"
        assert y is None, "y currently must be None"
        for idx, batch in enumerate(x):
            # Prepare input with constants
            x, y = prepare_batch_fno(
                batch, channels=self.channels, with_constants=False, with_time=True
            )  # type: ignore  # noqa: PGH003

            # Predictions
            y_pred = self.model(x)

            # Get loss
            # Take the first time idx as the next time step prediction
            loss = self.loss_fn(y_pred[:, :, :1, ...], y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if idx % 100 == 0:
                print(f"sample {idx:5d}, loss: {loss.item():.5e}")

    def forward(self, x: TensorLike):
        """Forward pass."""
        return self.model(x)[:, :, :1, :, :]

    def _predict(self, x: TensorLike | DataLoader, with_grad: bool) -> OutputLike:
        assert isinstance(x, DataLoader), "x currently must be a DataLoader"
        with torch.set_grad_enabled(with_grad):
            channels = (0,)  # Which channel to use
            all_preds = []
            for _, batch in enumerate(x):
                # Prepare input with constants
                x, _ = prepare_batch_fno(
                    batch, channels=channels, with_constants=False, with_time=True
                )
                out = self(x)
                all_preds.append(out)
            return torch.cat(all_preds)
