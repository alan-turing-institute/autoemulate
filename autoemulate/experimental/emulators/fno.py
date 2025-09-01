import torch
import torch.nn.functional as F
from autoemulate.core.types import TensorLike
from autoemulate.emulators.base import PyTorchBackend
from neuralop.models import TFNO
from torch.utils.data import DataLoader


def prepare_batch(sample, channels=(0,), with_constants=False, with_time=True):
    """Prepare a batch of input and output data."""
    # Get input fields, constant scalars and output fields
    x = sample["input_fields"][
        :, :, :, :, channels
    ]  # [batch, time, height, width, len(channels)]
    y = sample["output_fields"][
        :, :, :, :, channels
    ]  # [batch, time, height, width, len(channels)]
    # Permute both x and y
    x = x.permute(0, 4, 1, 2, 3)  # [batch, len(channels), time, height, width]
    y = y.permute(0, 4, 1, 2, 3)  # [batch, len(channels), time, height, width]

    # Only add constants to input, not output
    if with_constants:
        constant_scalars = sample["constant_scalars"]  # [batch, n_constants]

        # Assign spatio-temporal dims to constants
        time_window, height, width = x.shape[2], x.shape[3], x.shape[4]
        n_constants = constant_scalars.shape[-1]

        # Add spatio-temporal dims to constants
        c_broadcast = constant_scalars.reshape(-1, n_constants, 1, 1, 1).expand(
            -1, n_constants, time_window, height, width
        )

        # Concatenate along channel dimension
        x = torch.cat([x, c_broadcast], dim=1)

    if not with_time:
        # Take last time step for both input and output
        return x[:, :, -1:, :, :], y[:, :, -1:, :, :]  # Keep time dim as 1

    # Otherwise include full time
    return x, y


class FNOEmulator(PyTorchBackend):
    """A 5D FNO emulator for multivariable spatiotemporal modeling."""

    def __init__(
        self,
        x: TensorLike | None = None,  # noqa: ARG002
        y: TensorLike | None = None,  # noqa: ARG002
        n_vars: int = 2,
        channels: tuple[int, ...] = (0,),
        n_modes: tuple[int, int, int] = (16, 16, 1),  # Default to single timestep
        hidden_channels: int = 64,
        projection_channels: int = 128,
        n_layers: int = 4,
        use_skip_connections: bool = True,  # noqa: ARG002 I will use this later
        lifting_channels: int = 256,
        factorization: str = "tucker",
        rank: float = 0.42,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 16,
        **kwargs,
    ):
        """
        Initialize the 5D FNO emulator.

        Args:
            x, y: Training data (optional, following PyTorchBackend signature)
            n_vars: Number of physical variables (including constants if used)
            channels: Which channels to use during training
            n_modes: Fourier modes for (height, width, time)
            hidden_channels: Hidden dimension size
            projection_channels: Projection layer dimension
            n_layers: Number of Fourier layers
            use_skip_connections: Whether to use additional skip connections
            lifting_channels: Lifting layer dimension
            factorization: Type of tensor factorization
            rank: Rank for tensor factorization
            lr: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            **kwargs: Additional arguments
        """
        # Initialize nn.Module (required since PyTorchBackend inherits from nn.Module)
        super().__init__()

        # Store our custom parameters
        self.channels = channels
        self.n_modes = n_modes
        self.n_vars = n_vars

        # Store PyTorchBackend parameters (following the base class pattern)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        # Initialize the FNO model directly (following AutoEmulate pattern)
        self.model = TFNO(
            n_modes=n_modes,  # 3D modes: (height, width, time)
            hidden_channels=hidden_channels,
            in_channels=n_vars,
            out_channels=n_vars,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            use_mlp=True,
            mlp={"expansion": 0.5, "dropout": 0.0},
            non_linearity=F.gelu,
            fno_skip="linear",
            rank=rank,
            factorization=factorization,
            implementation="factorized",
            fft_norm="forward",
        )

        # Set up optimizer (required by PyTorchBackend)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def _fit(self, x: DataLoader, y: DataLoader | None = None):  # type: ignore # noqa: PGH003, ARG002
        """Fit the model to the training data."""
        for idx, batch in enumerate(x):
            # For full time series prediction, use with_time=True

            x_batch, y_batch = prepare_batch(
                batch,
                channels=self.channels,
                with_constants=False,
                with_time=True,  # This gives us full temporal sequences
            )  # type: ignore # noqa: PGH003
            # x_batch = x_batch.permute(0, 4, 1, 2, 3)  # [1, 1, 10, 64, 64]
            # y_batch = y_batch.permute(0, 4, 1, 2, 3)  # [1, 1, 58, 64, 64]

            # Move to device if available
            if hasattr(self, "device"):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

            # Predictions - now predicts full sequences
            y_pred = self.forward(x_batch)
            # Get loss - compare full predicted sequence with full target sequence
            loss = self.loss_fn(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            print(f"sample {idx:5d}, loss: {loss.item():.5e}")

    def forward(self, x: TensorLike) -> TensorLike:
        """Forward pass - required by PyTorchBackend."""
        return self.model(x)

    @staticmethod
    def is_multioutput() -> bool:
        """Flag to indicate if the model is multioutput or not."""
        return True  # MultivariableFNO handles multiple physical variables

    @staticmethod
    def get_tune_params():
        """Return hyperparameters to tune for time series prediction."""
        return {
            "hidden_channels": [32, 64, 128],
            "n_layers": [2, 4, 6],
            "lr": [1e-4, 1e-3, 1e-2],
            "batch_size": [4, 8, 16],  # Smaller batches for longer sequences
            # For time series, use more temporal modes
            "n_modes": [(12, 12, 8), (16, 16, 10), (20, 20, 12)],
            "factorization": ["tucker", "cp"],
            "rank": [0.1, 0.42, 0.8],
        }
