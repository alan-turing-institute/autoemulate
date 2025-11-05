from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import DeviceLike, GaussianLike, TensorLike, TuneParams
from autoemulate.emulators.base import DropoutTorchBackend, Emulator, GaussianEmulator
from autoemulate.emulators.nn.mlp import MLP
from autoemulate.transforms.standardize import StandardizeTransform
from autoemulate.transforms.utils import make_positive_definite


class Ensemble(GaussianEmulator):
    """
    Ensemble emulator that aggregates multiple Emulator instances to provide UQ.

    Ensemble emulator that aggregates multiple Emulator instances and returns
    a GaussianLike representing the ensemble posterior.
    Note that an Emulator instance may also be an Ensemble itself.
    """

    def __init__(
        self,
        emulators: Sequence[Emulator] | None = None,
        jitter: float = 1e-4,
        device: DeviceLike | None = None,
    ):
        """
        Initialize the Ensemble with a sequence of emulators.

        Parameters
        ----------
        emulators: Sequence[Emulator]
            A sequence of emulators to construct the ensemble with.
        jitter: float
            Amount of jitter to add to the covariance diagonal to avoid degeneracy.
            Defaults to 1e-4.
        device: DeviceLike | None
            The device to put torch tensors on.
        """
        assert isinstance(emulators, Sequence)
        for e in emulators:
            assert isinstance(e, Emulator)
        self.emulators = list(emulators)
        self.is_fitted_ = all(e.is_fitted_ for e in emulators)
        self.jitter = jitter
        self.supports_grad = all(e.supports_grad for e in self.emulators)
        TorchDeviceMixin.__init__(self, device=device)

    @staticmethod
    def is_multioutput() -> bool:
        """Ensemble supports multi-output."""
        return True

    @staticmethod
    def get_tune_params() -> TuneParams:
        """Return a dictionary of hyperparameters to tune."""
        return {}

    def _fit(self, x: TensorLike, y: TensorLike) -> None:
        for e in self.emulators:
            e.fit(x, y)
        self.is_fitted_ = True

    def _predict(self, x: Tensor, with_grad: bool) -> GaussianLike:
        if with_grad and not self.supports_grad:
            msg = "Gradient calculation is not supported."
            raise ValueError(msg)

        # Inference mode to disable autograd computation graph
        device = x.device
        means: list[Tensor] = []
        covs: list[Tensor] = []

        # Outputs from each emulator
        for e in self.emulators:
            out = e.predict(x, with_grad)
            if isinstance(out, GaussianLike):
                mu_i = out.mean.to(device)  # (batch_size, n_dims)
                assert isinstance(out.covariance_matrix, TensorLike)
                sigma_i = out.covariance_matrix.to(
                    device
                )  # (batch_size, n_dims, n_dims)
            elif isinstance(out, TensorLike):
                mu_i = out.to(device)
                # Instead of constructing dense zero matrix
                sigma_i = torch.broadcast_to(
                    torch.tensor(0.0),
                    (mu_i.shape[0], mu_i.shape[1], mu_i.shape[1]),
                ).to(device)
            else:
                s = f"Emulators of type {type(e)} are note supported yet."
                raise TypeError(s)
            means.append(mu_i)
            covs.append(sigma_i)

        # Stacked means and covs
        mu_stack = torch.stack(means, dim=0)  # (M, batch, dim)
        cov_stack = torch.stack(covs, dim=0)  # (M, batch, dim, dim)

        # Uniform ensemble mean and aleatoric average
        # NOTE: can implement weighting in future
        mu_ens = mu_stack.mean(dim=0)  # (batch, dim)
        sigma_alea = cov_stack.mean(dim=0)  # (batch, dim, dim)

        # Epistemic covariance: unbiased over M members
        dev = mu_stack - mu_ens.unsqueeze(0)  # (M, batch, dim)
        sigma_epi = torch.einsum("m b d, m b e -> b d e", dev, dev) / (
            len(self.emulators) - 1
        )

        # Total covariance
        sigma_ens = sigma_alea + sigma_epi  # (batch, dim, dim)

        # Return as MultivariateNormal
        return GaussianLike(
            mu_ens,
            make_positive_definite(
                sigma_ens, min_jitter=self.jitter, max_tries=3, clamp_eigvals=False
            ),
        )


class EnsembleMLP(Ensemble):
    """
    Ensemble of MLP emulators.

    This class is an ensemble of MLP emulators, each initialized with the same input and
    output data.

    """

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        standardize_x: bool = True,
        standardize_y: bool = True,
        n_emulators: int = 4,
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
        params_size: int = 1,
        random_seed: int | None = None,
        device: DeviceLike | None = None,
        scheduler_cls: type[LRScheduler] | None = None,
        scheduler_params: dict | None = None,
    ):
        """
        Initialize an ensemble of MLPs.

        Parameters
        ----------
        x: TensorLike
            Input data tensor of shape (batch_size, n_features).
        y: TensorLike
            Target values tensor of shape (batch_size, n_outputs).
        standardize_x: bool
            Whether to standardize the input data. Defaults to True.
        standardize_y: bool
            Whether to standardize the output data. Defaults to True.
        n_emulators: int
            Number of MLP emulators to create in the ensemble. Defaults to 4.
        device: DeviceLike | None
            Device to run the model on (e.g., "cpu", "cuda"). Defaults to None.
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
        params_size: int
            Number of parameters to predict per output dimension. Defaults to 1.
        random_seed: int | None
            Random seed for reproducibility. If None, no seed is set. Defaults to None.
        device: DeviceLike | None
            Device to run the model on (e.g., "cpu", "cuda", "mps"). Defaults to None.
        scheduler_cls: type[LRScheduler] | None
            Learning rate scheduler class. If None, no scheduler is used. Defaults to
            None.
        scheduler_params: dict | None
            Additional keyword arguments related to the scheduler.
        """
        emulators = [
            MLP(
                x,
                y,
                standardize_x=standardize_x,
                standardize_y=standardize_y,
                activation_cls=activation_cls,
                loss_fn_cls=loss_fn_cls,
                epochs=epochs,
                batch_size=batch_size,
                layer_dims=layer_dims,
                weight_init=weight_init,
                scale=scale,
                bias_init=bias_init,
                dropout_prob=dropout_prob,
                lr=lr,
                params_size=params_size,
                random_seed=random_seed,
                device=device,
                scheduler_cls=scheduler_cls,
                scheduler_params=scheduler_params,
            )
            for i in range(n_emulators)
        ]
        super().__init__(emulators, device=device)

    @staticmethod
    def is_multioutput() -> bool:
        """Ensemble of MLPs supports multi-output."""
        return True

    @staticmethod
    def get_tune_params() -> TuneParams:
        """Return a dictionary of hyperparameters to tune."""
        return {"n_emulators": [2, 4, 6, 8], **MLP.get_tune_params()}


class DropoutEnsemble(GaussianEmulator, TorchDeviceMixin):
    """
    Monte-Carlo Dropout ensemble.

    DropoutEnsemble does a number of forward passes with dropout on, and computes mean
    and epistemic covariance across them.
    """

    def __init__(
        self,
        model: DropoutTorchBackend,
        standardize_x: bool = True,
        standardize_y: bool = True,
        n_samples: int = 20,
        jitter: float = 1e-4,
        device: DeviceLike | None = None,
    ):
        """
        Initialize the DropoutEnsemble with a fitted model.

        Parameters
        ----------
        model: DropoutTorchBackend
            A fitted DropoutTorchBackend (or any nn.Module with dropout layers).
        n_samples: int
            Number of forward passes to perform.
        jitter: float
            Amount of jitter to add to covariance diagonal to avoide degeneracy.
        device: DeviceLike | None
            torch device for inference (e.g. "cpu", "cuda").
        """
        assert isinstance(model, DropoutTorchBackend), "model must be a PyTorchBackend"
        TorchDeviceMixin.__init__(self, device=device)
        assert n_samples > 0
        self.model = model.to(self.device)
        self.x_transform = StandardizeTransform() if standardize_x else None
        self.y_transform = StandardizeTransform() if standardize_y else None
        self.n_samples = n_samples
        self.is_fitted_ = model.is_fitted_
        self.jitter = jitter
        self.supports_grad = True

    @staticmethod
    def is_multioutput() -> bool:
        """DropoutEnsemble supports multi-output."""
        return True

    @staticmethod
    def get_tune_params() -> TuneParams:
        """Return a dictionary of hyperparameters to tune."""
        return {
            "n_samples": [10, 20, 50, 100],
        }

    def _fit(self, x: TensorLike, y: TensorLike) -> None:
        # Delegate training to the wrapped model
        self.model.fit(x, y)
        self.is_fitted_ = True

    def _predict(self, x: Tensor, with_grad: bool) -> GaussianLike:
        if not self.is_fitted_:
            s = "DropoutEnsemble: model is not fitted yet."
            raise RuntimeError(s)

        # move input to right device
        x = x.to(self.device)

        # enable dropout
        self.model.train()

        # collect M outputs
        samples = []
        with torch.set_grad_enabled(with_grad):
            for _ in range(self.n_samples):
                out = self.model.forward(x)
                # out: Tensor of shape (batch_size, output_dim)
                samples.append(out)

        # stack to shape (M, batch, dim)
        stack = torch.stack(samples, dim=0)

        # ensemble mean: (batch, dim)
        mu = stack.mean(dim=0)

        # compute epistemic covariance for each batch point
        # dev shape (M, batch, dim)
        dev = stack - mu.unsqueeze(0)
        # sigma_epi: (batch, dim, dim)
        sigma_epi = torch.einsum("m b d, m b e -> b d e", dev, dev) / (
            self.n_samples - 1
        )

        return GaussianLike(
            mu,
            make_positive_definite(
                sigma_epi, min_jitter=self.jitter, max_tries=3, clamp_eigvals=False
            ),
        )


class EnsembleMLPDropout(DropoutEnsemble):
    """Ensemble of MLP emulators with dropout."""

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        standardize_x: bool = True,
        standardize_y: bool = True,
        activation_cls: type[nn.Module] = nn.ReLU,
        loss_fn_cls: type[nn.Module] = nn.MSELoss,
        epochs: int = 100,
        batch_size: int = 16,
        layer_dims: list[int] | None = None,
        weight_init: str = "default",
        scale: float = 1.0,
        bias_init: str = "default",
        dropout_prob: float | None = 0.2,
        lr: float = 1e-2,
        params_size: int = 1,
        random_seed: int | None = None,
        device: DeviceLike | None = None,
        scheduler_cls: type[LRScheduler] | None = None,
        scheduler_params: dict | None = None,
    ):
        """
        Initialize an ensemble of MLPs with dropout.

        Parameters
        ----------
        x: TensorLike
            Input data tensor of shape (batch_size, n_features).
        y: TensorLike
            Target values tensor of shape (batch_size, n_outputs).
        standardize_x: bool
            Whether to standardize the input data. Defaults to True.
        standardize_y: bool
            Whether to standardize the output data. Defaults to True.
        device: DeviceLike | None
            Device to run the model on (e.g., "cpu", "cuda"). Defaults to None.
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
            Defaults to 0.2.
        lr: float
            Learning rate for the optimizer. Defaults to 1e-2.
        params_size: int
            Number of parameters to predict per output dimension. Defaults to 1.
        random_seed: int | None
            Random seed for reproducibility. If None, no seed is set. Defaults to None.
        device: DeviceLike | None
            Device to run the model on (e.g., "cpu", "cuda", "mps"). Defaults to None.
        scheduler_cls: type[LRScheduler] | None
            Learning rate scheduler class. If None, no scheduler is used. Defaults to
            None.
        scheduler_params: dict | None
            Additional keyword arguments related to the scheduler.

        """
        DropoutEnsemble.__init__(
            self,
            MLP(
                x,
                y,
                standardize_x=standardize_x,
                standardize_y=standardize_y,
                activation_cls=activation_cls,
                loss_fn_cls=loss_fn_cls,
                epochs=epochs,
                batch_size=batch_size,
                layer_dims=layer_dims,
                weight_init=weight_init,
                scale=scale,
                bias_init=bias_init,
                dropout_prob=dropout_prob,
                lr=lr,
                params_size=params_size,
                random_seed=random_seed,
                device=device,
                scheduler_cls=scheduler_cls,
                scheduler_params=scheduler_params,
            ),
            device=device,
        )

    @staticmethod
    def get_tune_params() -> TuneParams:
        """Return a dictionary of hyperparameters to tune."""
        params = MLP.get_tune_params()
        params["dropout_prob"] = [el for el in params["dropout_prob"] if el is not None]
        return {"n_emulators": [2, 4, 6, 8], **params}
