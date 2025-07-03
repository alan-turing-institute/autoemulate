from collections.abc import Sequence

import torch
from torch import Tensor

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import (
    Emulator,
    GaussianEmulator,
    PyTorchBackend,
)
from autoemulate.experimental.types import (
    DeviceLike,
    GaussianLike,
    TensorLike,
    TuneConfig,
)


class Ensemble(GaussianEmulator):
    """
    Ensemble emulator that aggregates multiple Emulator instances and returns
    a MultivariateNormal representing the ensemble posterior.
    Note that an Emulator instance may also be an Ensemble itself.

    Parameters
        Emulators: list[Emulator] List of fitted emulator instances.
        Each member's `predict(x: Tensor)`
        must return either:
            - TensorLike (treated as mean)
            - MultivariateNormal (with `.mean` and `.covariance_matrix`)
    """

    def __init__(
        self,
        emulators: list[Emulator],
        device: DeviceLike | None = None,
    ):
        assert isinstance(emulators, Sequence)
        for e in emulators:
            assert isinstance(e, Emulator)
        self.emulators = list(emulators)
        self.is_fitted_ = all(e.is_fitted_ for e in emulators)
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
        """
        Perform inference with the ensemble.

        Inputs:
            - x of shape (batch_size, n_dims)

        Returns:
            - torch.distributions.MultivariateNormal with
                - mean of shape (batch_size, n_dims)
                - cov of shape (batch_size, n_dims, n_dims)
        """

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
                    torch.tensor(0.0), (mu_i.shape[0], mu_i.shape[1], mu_i.shape[1])
                )
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
        return GaussianLike(mu_ens, sigma_ens)


class DropoutEnsemble(GaussianEmulator, TorchDeviceMixin):
    """
    Monte-Carlo Dropout ensemble: do N forward passes with dropout on,
    and compute mean + (epistemic) covariance across them.
    """

    def __init__(
        self,
        model: PyTorchBackend,
        n_samples: int = 20,
        jitter: float = 1e-6,
        device: DeviceLike | None = None,
    ):
        """
        Parameters
        ----------
        model
            A fitted PyTorchBackend (or any nn.Module with dropout layers).
        n_samples
            Number of stochastic forward passes to perform.
        device
            torch device for inference (e.g. "cpu", "cuda").
        """
        assert isinstance(model, PyTorchBackend), "model must be a PyTorchBackend"
        TorchDeviceMixin.__init__(self, device=device)

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

        # Add some jitter to avoid positive-definite warnings
        b, d = mu.shape
        eye = torch.eye(d, device=sigma_epi.device)  # (dim, dim)
        jitter_mat = eye.unsqueeze(0).expand(b, d, d) * self.jitter
        sigma_epi += jitter_mat

        return GaussianLike(mu, sigma_epi)
