from torch import nn, optim

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.types import DeviceLike, TensorLike

from ..base import PyTorchBackend


class MLP(PyTorchBackend):
    def __init__(  # noqa: PLR0913
        self,
        x: TensorLike,
        y: TensorLike,
        activation_cls: type[nn.Module] = nn.ReLU,
        layer_dims: list[int] | None = None,
        weight_init: str = "default",
        scale: float = 1.0,
        dropout_prob: float | None = None,
        loss_fn: type[nn.Module] = nn.MSELoss,
        lr: float = 1e-1,
        weight_decay: float = 0.0,
        device: DeviceLike | None = None,
        **kwargs,
    ):
        TorchDeviceMixin.__init__(self, device=device)
        nn.Module.__init__(self)
        x, y = self._convert_to_tensors(x, y)
        self.dropout_prob = dropout_prob

        # Construct the MLP layers
        layer_dims = [x.shape[1], *layer_dims] if layer_dims else [x.shape[1], 32, 16]
        layers = []
        for idx, dim in enumerate(layer_dims[1:]):
            layers.append(nn.Linear(layer_dims[idx], dim, device=self.device))
            layers.append(activation_cls())
            if self.dropout_prob is not None:
                layers.append(nn.Dropout(p=self.dropout_prob))
        # Add final layer without activation
        layers.append(nn.Linear(layer_dims[-1], y.shape[1], device=self.device))
        self.nn = nn.Sequential(*layers)

        self._initialize_weights(weight_init, scale)
        self.set_loss_function(loss_fn())
        self.set_optimizer(
            optim.Adam(self.nn.parameters(), lr=lr, weight_decay=weight_decay)
        )
        self.to(device)

    def forward(self, x):
        return self.nn(x)

    @staticmethod
    def is_multioutput() -> bool:
        return True

    @staticmethod
    def get_tune_config():
        return {
            "epochs": [50, 100, 200],
            "layer_dims": [[32, 16], [64, 32, 16]],
            "lr": [1e-1, 1e-2, 1e-3],
            "weight_decay": [0.0, 1e-4, 1e-3],
            "batch_size": [16, 32],
            "weight_init": ["default", "normal"],
            "scale": [0.1, 1.0],
            "bias_init": ["default", "zeros"],
            "dropout_prob": [0.3, 0.5, None],
        }
