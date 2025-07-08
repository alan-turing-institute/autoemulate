from collections.abc import Sequence

import torch
from torch import Tensor

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import (
    DropoutTorchBackend,
    Emulator,
    GaussianEmulator,
)
from autoemulate.experimental.emulators.nn.mlp import MLP
from autoemulate.experimental.transforms.utils import make_positive_definite
from autoemulate.experimental.types import (
    DeviceLike,
    GaussianLike,
    TensorLike,
    TuneConfig,
)


class Ensemble(GaussianEmulator):
    """
    Ensemble emulator that aggregates multiple Emulator instances and returns
    a GaussianLike representing the ensemble posterior.
    Note that an Emulator instance may also be an Ensemble itself.
    """

    def __init__(
        self,
        emulators: Sequence[Emulator] | None = None,
        jitter: float = 1e-6,
        device: DeviceLike | None = None,
    ):
        """
        Parameters
        ----------
        emulators: Sequence[Emulator]
            A sequence of emulators to construct the ensemble with.
        jitter: float, default=1e-6
            Amount of jitter to add to the covariance diagonal to avoid degeneracy.
        device: DeviceLike | None
            The device to put torch tensors on.
        """

        assert isinstance(emulators, Sequence)
        for e in emulators:
            assert isinstance(e, Emulator)
        self.emulators = list(emulators)
        self.is_fitted_ = all(e.is_fitted_ for e in emulators)
        self.jitter = jitter
        TorchDeviceMixin.__init__(self, device=device)

    @staticmethod
    def is_multioutput() -> bool:
        return True

    @staticmethod
    def get_tune_config() -> TuneConfig:
        return {}

    def _fit(self, x: TensorLike, y: TensorLike) -> None:
        for e in self.emulators:
            e.fit(x, y)
        self.is_fitted_ = True

    @torch.inference_mode()
    def _predict(self, x: Tensor) -> GaussianLike:
        # Inference mode to disable autograd computation graph
        device = x.device
        means: list[Tensor] = []
        covs: list[Tensor] = []

        # Outputs from each emulator
        for e in self.emulators:
            out = e.predict(x)
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
                sigma_ens, max_tries_epsilon=2, max_tries_min_eigval=0
            ),
        )


class EnsembleMLP(Ensemble):
    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        n_emulators: int = 4,
        device: DeviceLike | None = None,
        **mlp_kwargs,
    ):
        """Initialize an ensemble of MLPs.

        Parameters
        ----------
        x : TensorLike
            Input data tensor of shape (batch_size, n_features).
        y : TensorLike
            Target values tensor of shape (batch_size, n_outputs).
        n_emulators : int, default=4
            Number of MLP emulators to create in the ensemble.
        device : DeviceLike | None, default=None
            Device to run the model on (e.g., "cpu", "cuda").
        **mlp_kwargs : dict
            Additional keyword arguments for the MLP constructor.

        """
        emulators = [
            MLP(x, y, random_seed=i, device=device, **mlp_kwargs)
            for i in range(n_emulators)
        ]
        super().__init__(emulators, device=device)

    @staticmethod
    def get_tune_config() -> TuneConfig:
        return {"n_emulators": [2, 4, 6, 8], **MLP.get_tune_config()}


class DropoutEnsemble(GaussianEmulator, TorchDeviceMixin):
    """
    Monte-Carlo Dropout ensemble: do a number of forward passes with dropout on,
    and compute mean + epistemic covariance across them.
    """

    def __init__(
        self,
        model: DropoutTorchBackend,
        n_samples: int = 20,
        jitter: float = 1e-6,
        device: DeviceLike | None = None,
    ):
        """
        Parameters
        ----------
        model : PyTorchBackend
            A fitted PyTorchBackend (or any nn.Module with dropout layers).
        n_samples : int
            Number of forward passes to perform.
        jitter : float
            Amount of jitter to add to covariance diagonal to avoide degeneracy.
        device : DeviceLike | None
            torch device for inference (e.g. "cpu", "cuda").
        """
        assert isinstance(model, DropoutTorchBackend), "model must be a PyTorchBackend"
        TorchDeviceMixin.__init__(self, device=device)
        assert n_samples > 0
        self.model = model.to(self.device)
        self.n_samples = n_samples
        self.is_fitted_ = model.is_fitted_
        self.jitter = jitter

    @staticmethod
    def is_multioutput() -> bool:
        return True

    @staticmethod
    def get_tune_config() -> TuneConfig:
        return {
            "n_samples": [10, 20, 50, 100],
        }

    def _fit(self, x: TensorLike, y: TensorLike) -> None:
        # Delegate training to the wrapped model
        self.model.fit(x, y)
        self.is_fitted_ = True

    @torch.inference_mode()
    def _predict(self, x: Tensor) -> GaussianLike:
        if not self.is_fitted_:
            s = "DropoutEnsemble: model is not fitted yet."
            raise RuntimeError(s)

        # move input to right device
        x = x.to(self.device)

        # enable dropout
        self.model.train()

        # collect M outputs
        samples = []
        for _ in range(self.n_samples):
            # apply any preprocessing the model expects
            x_proc = self.model.preprocess(x)
            out = self.model.forward(x_proc)
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
                sigma_epi, max_tries_epsilon=2, max_tries_min_eigval=0
            ),
        )


class EnsembleMLPDropout(DropoutEnsemble):
    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        dropout_prob: float = 0.2,
        device: DeviceLike | None = None,
        **mlp_kwargs,
    ):
        """Initialize an ensemble of MLPs with dropout.

        Parameters
        ----------
        x : TensorLike
            Input data tensor of shape (batch_size, n_features).
        y : TensorLike
            Target values tensor of shape (batch_size, n_outputs).
        dropout_prob : float, default=0.2
            Dropout probability to use in the MLP layers.
        device : DeviceLike | None, default=None
            Device to run the model on (e.g., "cpu", "cuda").
        **mlp_kwargs : dict
            Additional keyword arguments for the MLP constructor.

        """
        super().__init__(
            MLP(x, y, dropout_prob=dropout_prob, device=device, **mlp_kwargs),
            device=device,
        )

    @staticmethod
    def get_tune_config() -> TuneConfig:
        config = MLP.get_tune_config()
        config["dropout_prob"] = [el for el in config["dropout_prob"] if el is not None]
        return {"n_emulators": [2, 4, 6, 8], **config}
