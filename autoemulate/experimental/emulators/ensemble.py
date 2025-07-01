import torch
from torch import Tensor

from autoemulate.experimental.data.utils import ValidationMixin
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import Emulator, GaussianEmulator, DeterministicEmulator, PyTorchBackend
from autoemulate.experimental.types import DeviceLike, TensorLike, TuneConfig, GaussianLike
from typing import Sequence


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
        emulators: list[GaussianEmulator | DeterministicEmulator | PyTorchBackend], 
        device: DeviceLike | None = None
    ):
        assert isinstance(emulators, Sequence)
        for e in emulators:
            assert isinstance(e, GaussianEmulator | DeterministicEmulator | PyTorchBackend)
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
                mu_i = out.mean.to(device) # (batch_size, n_dims)
                sigma_i = out.covariance_matrix.to(device) # (batch_size, n_dims, n_dims)
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
        sigma_epi = torch.einsum("m b d, m b e -> b d e", dev, dev) / (len(self.emulators) - 1)

        # Total covariance
        sigma_ens = sigma_alea + sigma_epi  # (batch, dim, dim)

        # Return as MultivariateNormal
        return GaussianLike(mu_ens, sigma_ens)
