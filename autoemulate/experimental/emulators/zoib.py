import torch
import torch.nn.functional as F
from autoemulate.core.types import DeviceLike, TensorLike
from autoemulate.emulators.nn.mlp import MLP
from torch import nn


class ZeroOneInflatedBeta(torch.distributions.Distribution):
    """ZeroOneInflatedBeta."""

    arg_constraints = {  # type: ignore  # noqa: PGH003, RUF012
        "pi0": torch.distributions.constraints.unit_interval,  # type: ignore  # noqa: PGH003
        "pi1": torch.distributions.constraints.unit_interval,  # type: ignore  # noqa: PGH003
        "concentration1": torch.distributions.constraints.positive,  # type: ignore  # noqa: PGH003
        "concentration0": torch.distributions.constraints.positive,  # type: ignore  # noqa: PGH003
    }
    support = torch.distributions.constraints.unit_interval  # type: ignore  # noqa: PGH003

    def __init__(self, pi0, pi1, concentration1, concentration0, validate_args=None):
        self.pi0 = pi0
        self.pi1 = pi1
        self.concentration1 = concentration1
        self.concentration0 = concentration0
        self.beta = torch.distributions.Beta(concentration1, concentration0)

        # Ensure pi0 + pi1 < 1 elementwise
        if ((self.pi0 + self.pi1) > torch.ones_like(self.pi0)).any():
            msg = "pi0 + pi1 must be <= 1"
            raise ValueError(msg)

        super().__init__(validate_args=validate_args)

    def log_prob(self, value):
        """Log prob."""
        EPS = 1e-12
        # Ensure value can broadcast with parameters. If value is 1D [N], make it [N, 1]
        # to align with [N, num_tasks].
        value_in = value
        squeeze_back = False
        while value_in.dim() < self.pi0.dim():
            value_in = value_in.unsqueeze(-1)
            squeeze_back = True

        # Clamp continuous values away from boundaries for Beta support
        v = value_in.clamp(EPS, 1 - EPS)

        # Broadcast pi0, pi1 to value shape
        pi0_b = self.pi0.expand_as(value_in)
        pi1_b = self.pi1.expand_as(value_in)

        # Mixture log probs
        logp0 = torch.log(pi0_b + EPS)
        logp1 = torch.log(pi1_b + EPS)
        mix_log = torch.log(1 - pi0_b - pi1_b + EPS)
        beta_lp = self.beta.log_prob(v)

        cont = mix_log + beta_lp
        logp = torch.where(
            value_in == 0, logp0, torch.where(value_in == 1, logp1, cont)
        )

        if squeeze_back and logp.shape[-1] == 1:
            logp = logp.squeeze(-1)
        return logp

    @property
    def mean(self):
        """Mixture mean: pi1*1 + (1-pi0-pi1)*beta.mean."""
        p0 = self.pi0
        p1 = self.pi1
        p_cont = 1 - p0 - p1
        return p1 + p_cont * self.beta.mean

    @property
    def variance(self):
        """Mixture variance computed from mixture second moment."""
        p0 = self.pi0
        p1 = self.pi1
        p_cont = 1 - p0 - p1
        # E[X] for mixture
        mean = p1 + p_cont * self.beta.mean
        # E[X^2] for Beta is var + mean^2
        beta_second = self.beta.variance + self.beta.mean**2
        second_moment = p1 + p_cont * beta_second
        return second_moment - mean**2

    def sample(self, sample_shape=None):
        """Sample."""
        # Sample from categorical: {0, 1, Beta}
        if sample_shape is None:
            sample_shape = torch.Size()
        ones = torch.ones_like(self.pi0)
        probs = torch.stack([self.pi0, self.pi1, ones - self.pi0 - self.pi1], dim=-1)
        cat = torch.distributions.Categorical(probs=probs)
        choice = cat.sample(sample_shape)

        beta_samples = self.beta.sample(sample_shape)

        # Assign values based on choice
        return torch.where(
            choice == 0,
            torch.zeros_like(beta_samples),
            torch.where(choice == 1, torch.ones_like(beta_samples), beta_samples),
        )


class ZOIBMLP(MLP):
    """
    Zero-One Inflated Beta distribution Multi-Layer Perceptron (MLP) emulator.

    MLP provides a simple deterministic emulator with optional model stochasticity
    provided by different weight initialization and dropout.
    """

    supports_uq: bool = True

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        standardize_x: bool = True,
        activation_cls: type[nn.Module] = nn.ReLU,
        loss_fn_cls: type[nn.Module] = nn.MSELoss,
        epochs: int = 100,
        batch_size: int = 16,
        layer_dims: list[int] | None = None,
        weight_init: str = "default",
        scale: float = 1.0,
        bias_init: str = "default",
        dropout_prob: float | None = None,
        lr: float = 1e-2,
        random_seed: int | None = None,
        device: DeviceLike | None = None,
        **scheduler_kwargs,
    ):
        """
        Zero-One Inflated Beta Distribution Multi-Layer Perceptron (MLP) emulator.

        Parameters
        ----------
        x: TensorLike
            Input features.
        y: TensorLike
            Target values.
        activation_cls: type[nn.Module]
            Activation function to use in the hidden layers. Defaults to `nn.ReLU`.
        layer_dims: list[int] | None
            Dimensions of the hidden layers. If None, defaults to [32, 16].
            Defaults to None.
        weight_init: str
            Weight initialization method. Options are "default", "normal", "uniform",
            "zeros", "ones", "xavier_uniform", "xavier_normal", "kaiming_uniform",
            "kaiming_normal". Defaults to "default".
        scale: float
            Scale parameter for weight initialization methods. Used as:
            - gain for Xavier methods
            - std for normal distribution
            - bound for uniform distribution (range: [-scale, scale])
            - ignored for Kaiming methods (uses optimal scaling)
            Defaults to 1.0.
        bias_init: str
            Bias initialization method. Options: "zeros", "default":
                - "zeros" initializes biases to zero
                - "default" uses PyTorch's default uniform initialization
        dropout_prob: float | None
            Dropout probability for regularization. If None, no dropout is applied.
            Defaults to None.
        lr: float
            Learning rate for the optimizer. Defaults to 1e-2.
        random_seed: int | None
            Random seed for reproducibility. If None, no seed is set. Defaults to None.
        device: DeviceLike | None
            Device to run the model on (e.g., "cpu", "cuda", "mps"). Defaults to None.
        **scheduler_kwargs: dict
            Additional keyword arguments related to the scheduler.

        Raises
        ------
        ValueError
            If the input dimensions of `x` and `y` are not matrices.
        """
        MLP.__init__(
            self,
            x,
            y,
            standardize_x,
            False,  # Don't standardize y for ZOIB
            activation_cls,
            loss_fn_cls,
            epochs,
            batch_size,
            layer_dims,
            weight_init,
            scale,
            bias_init,
            dropout_prob,
            lr,
            5,  # params_size=5 for Zero-Inflated Beta distribution
            random_seed,
            device,
            **scheduler_kwargs,
        )

    def loss_func(self, y_pred, y_true):  # noqa: D102
        return -y_pred.log_prob(y_true).mean()

    def forward(self, x: TensorLike) -> ZeroOneInflatedBeta:
        """Forward pass for the MLP."""
        EPS = 1e-6
        output = self.nn(x)
        probs = F.softmax(output[..., 2:5], dim=-1)
        return ZeroOneInflatedBeta(
            pi0=probs[..., :1],
            pi1=probs[..., 1:2],
            concentration0=F.softplus(output[..., :1]) + EPS,
            concentration1=F.softplus(output[..., 1:2]) + EPS,
        )

    def predict_mean_and_variance(
        self,
        x: TensorLike,
        with_grad: bool = False,
        n_samples: int = 1000,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the mean and variance of the output for given input.

        Parameters
        ----------
        x: TensorLike
            Input features as numpy array or PyTorch tensor.

        Returns
        -------
        mean: torch.Tensor
            Predicted mean values.
        variance: torch.Tensor
            Predicted variance values.
        """
        self.eval()  # Set model to evaluation mode
        with torch.set_grad_enabled(with_grad):
            beta_dist = self.predict(x)
            assert isinstance(beta_dist, ZeroOneInflatedBeta)
            return beta_dist.mean, beta_dist.variance
