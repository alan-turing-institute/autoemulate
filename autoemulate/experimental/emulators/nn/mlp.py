from torch import nn

from autoemulate.experimental.data.utils import set_random_seed
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.types import DeviceLike, TensorLike

from ..base import DropoutTorchBackend


class MLP(DropoutTorchBackend):
    def __init__(  # noqa: PLR0913
        self,
        x: TensorLike,
        y: TensorLike,
        activation_cls: type[nn.Module] = nn.ReLU,
        loss_fn_cls: type[nn.Module] = nn.MSELoss,
        epochs: int = 100,
        layer_dims: list[int] | None = None,
        weight_init: str = "default",
        scale: float = 1.0,
        bias_init: str = "default",
        dropout_prob: float | None = None,
        lr: float = 1e-2,
        random_seed: int | None = None,
        device: DeviceLike | None = None,
        **kwargs,
    ):
        """
        Multi-Layer Perceptron (MLP) emulator.

        MLP provides a simple deterministic emulator with optional model stochasticity
        provided by different weight initialization and dropout.

        Parameters
        ----------
        x: TensorLike
            Input features.
        y: TensorLike
            Target values.
        activation_cls: type[nn.Module], default=nn.ReLU
            Activation function to use in the hidden layers.
        layer_dims: list[int] | None, default=None
            Dimensions of the hidden layers. If None, defaults to [32, 16].
        weight_init: str, default="default"
            Weight initialization method. Options are "default", "normal", "uniform",
            "zeros", "ones", "xavier_uniform", "xavier_normal", "kaiming_uniform",
            "kaiming_normal".
        scale: float, default=1.0
            Scale parameter for weight initialization methods. Used as:
            - gain for Xavier methods
            - std for normal distribution
            - bound for uniform distribution (range: [-scale, scale])
            - ignored for Kaiming methods (uses optimal scaling)
        bias_init: str
            Bias initialization method. Options: "zeros", "default":
                - "zeros" initializes biases to zero
                - "default" uses PyTorch's default uniform initialization
        dropout_prob: float | None, default=None
            Dropout probability for regularization. If None, no dropout is applied.
        lr: float, default=1e-1
            Learning rate for the optimizer.
        random_seed: int | None
            Random seed for reproducibility. If None, no seed is set.
        device: DeviceLike | None, default=None
            Device to run the model on (e.g., "cpu", "cuda", "mps")
        **kwargs: dict
            Additional keyword arguments.

        Raises
        ------
        ValueError
            If the input dimensions of `x` and `y` are not matrices.
        """
        TorchDeviceMixin.__init__(self, device=device)
        nn.Module.__init__(self)

        if random_seed is not None:
            set_random_seed(seed=random_seed)

        # Ensure x and y are tensors with correct dimensions
        x, y = self._convert_to_tensors(x, y)

        # Construct the MLP layers
        layer_dims = [x.shape[1], *layer_dims] if layer_dims else [x.shape[1], 32, 16]
        layers = []
        for idx, dim in enumerate(layer_dims[1:]):
            layers.append(nn.Linear(layer_dims[idx], dim, device=self.device))
            layers.append(activation_cls())
            if dropout_prob is not None:
                layers.append(nn.Dropout(p=dropout_prob))

        # Add final layer without activation
        num_tasks = y.shape[1]
        layers.append(nn.Linear(layer_dims[-1], num_tasks, device=self.device))
        self.nn = nn.Sequential(*layers)

        # Finalize initialization
        self._initialize_weights(weight_init, scale, bias_init)
        self.epochs = epochs
        self.loss_fn = loss_fn_cls()
        self.lr = lr
        self.optimizer = self.optimizer_cls(self.nn.parameters(), lr=self.lr)  # type: ignore[call-arg] since all optimizers include lr
        self.scheduler_setup(kwargs)
        self.to(device)

    def forward(self, x):
        return self.nn(x)

    @staticmethod
    def is_multioutput() -> bool:
        return True

    @staticmethod
    def get_tune_config():
        scheduler_params = MLP.scheduler_config()
        return {
            # "epochs": [50, 100, 200],
            "epochs": [100, 200],
            # "layer_dims": [[32, 16], [64, 32, 16]],
            "layer_dims": [[8, 4], [16, 8], [32, 16]],
            # "lr": [5e-1, 2e-1, 1e-1, 1e-2, 1e-3],
            "lr": [5e-1, 2e-1, 1e-1, 1e-2, 1e-3],
            "batch_size": [16, 32],
            "weight_init": ["default", "normal"],
            "scale": [0.1, 1.0],
            "bias_init": ["default", "zeros"],
            "dropout_prob": [0.3, None],
            "scheduler_cls": scheduler_params["scheduler_cls"],
            "scheduler_kwargs": scheduler_params["scheduler_kwargs"],
        }
