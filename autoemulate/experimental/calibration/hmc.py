import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.types import DeviceLike, GaussianLike, TensorLike


class HMCCalibrator(TorchDeviceMixin):
    def __init__(
        self, observations, emulator: Emulator, device: DeviceLike | None = None
    ):
        """ """
        TorchDeviceMixin.__init__(self, device=device)
        self.observations = observations
        self.emulator = emulator

    def model(self):
        """Pyro model for MCMC calibration..."""
        pass

    def run_mcmc(
        self, num_samples: int = 1000, warmup_steps: int = 500, num_chains: int = 1
    ):
        """Run MCMC sampling..."""

        nuts_kernel = NUTS(self.model)

        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains,
            # initial_params=initial_params,
        )

        mcmc.run()
