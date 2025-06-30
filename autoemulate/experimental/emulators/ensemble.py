import torch
from torch import Tensor

from autoemulate.experimental.data.utils import ValidationMixin
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.types import DeviceLike, TensorLike, TuneConfig


class Ensemble(Emulator, ValidationMixin, TorchDeviceMixin):
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

    def __init__(self, emulators: list[Emulator], device: DeviceLike | None = None):
        if not emulators:
            s = "Ensemble must contain at least one emulator."
            raise ValueError(s)
        self.emulators = emulators
        self.M = len(emulators)
        self.is_fitted_ = all(getattr(em, "is_fitted_", False) for em in emulators)
        TorchDeviceMixin.__init__(self, device=device)

    @staticmethod
    def is_multioutput() -> bool:
        return True

    @staticmethod
    def get_tune_config() -> TuneConfig:
        # No tunable hyperparameters at the ensemble level
        return {}

    def _fit(self, x: TensorLike, y: TensorLike) -> None:
        for em in self.emulators:
            em.fit(x, y)
        self.is_fitted_ = True

    def _predict(self, x: Tensor) -> torch.distributions.MultivariateNormal:
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
        with torch.inference_mode():
            device = x.device
            means: list[Tensor] = []
            covs: list[Tensor] = []

            # Outputs from each emulator
            for em in self.emulators:
                out = em.predict(x)
                if isinstance(out, torch.distributions.MultivariateNormal):
                    mu_i = out.mean.to(device)
                    sigma_i = out.covariance_matrix.to(device)
                elif isinstance(out, TensorLike):
                    mu_i = out.to(device)
                    dim = mu_i.shape[-1]
                    sigma_i = torch.zeros(mu_i.size(0), dim, dim, device=device)
                else:
                    s = (
                        "Sub-emulators' output must "
                        "be TensorLike or MultivariateNormal."
                    )
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
            sigma_epi = torch.einsum("m b d, m b e -> b d e", dev, dev) / (self.M - 1)

            # Total covariance
            sigma_ens = sigma_alea + sigma_epi  # (batch, dim, dim)

            # Return as MultivariateNormal
            return torch.distributions.MultivariateNormal(
                loc=mu_ens, covariance_matrix=sigma_ens
            )
