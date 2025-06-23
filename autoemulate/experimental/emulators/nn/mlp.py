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
        device: DeviceLike | None = None,
        **kwargs,
    ):
        TorchDeviceMixin.__init__(self, device=device)
        nn.Module.__init__(self)
        x, y = self._convert_to_tensors(x, y)
        layer_dims = layer_dims or [32, 16]
        layer_dims = [x.shape[1], *layer_dims]

        # Construct the MLP layers
        layers = []
        for idx, dim in enumerate(layer_dims[1:]):
            layers.append(nn.Linear(layer_dims[idx], dim))
            layers.append(activation_cls())
        layers.append(nn.Linear(layer_dims[-1], y.shape[1]))
        self.nn = nn.Sequential(*layers)

        # Init weights using backend specific method
        self._initialize_weights(weight_init, scale)

        # TODO: consider adding flexibility over optimizer to API
        self.optimizer = optim.Adam(
            self.nn.parameters(),
            lr=kwargs.get("lr", 1e-3),
            weight_decay=kwargs.get("weight_decay", 0.0),
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
            "lr": [1e-1, 1e-2, 1e-3, 1e-4],
            "batch_size": [16, 32],
            "weight_init": ["default", "normal"],
            "scale": [0.1, 1.0],
            "bias_init": [
                "default",
                "zeros",
            ],
        }
