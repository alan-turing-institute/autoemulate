import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.types import DeviceLike, DistributionLike, TensorLike


class HMCCalibrator(TorchDeviceMixin):
    """
    Markov Chain Monte Carlo (MCMC) is a Bayesian calibration method that estimates the
    probability distribution over input parameters given observed data. A key advantage
    is that it provides uncertainty estimates over the parameter space.

    Hamiltonian Monte Carlo (HMC) is a type of MCMC, which efficiently handles high
    dimensional parameter spaces. In particular, we use the NUTS sampler.
    """

    def __init__(  # noqa: PLR0913
        self,
        emulator: Emulator,
        parameter_range: dict[str, list[float]],
        observations: dict[str, float] | dict[str, list[float]],
        observation_noise: float | dict[str, float] = 0.1,
        calibration_params: list[str] | None = None,
        device: DeviceLike | None = None,
    ):
        """
        Initialize the HMC calibration object.

        Parameters
        ----------
        emulator: Emulator
            Fitted Emulator object.
        parameters_range : dict[str, tuple[float, float]]
            Dictionary mapping input parameter names to their (min, max) ranges.
        observations: dict[str, float] | dict[str, list[float]]
            A dictionary of either a single value or a list of values per output.
        observation_noise: float | dict[str, float]
            A single value or a dictionary of values (one per output). Defaults to 0.1.
        calibration_params: list[str] | None
            Optional list of input parameters to calibrate. Any parameters that are not
            listed will be set to the midpoint value of their parameter range. If None,
            will calibrate all input parameters. Defaults to None.
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.

        Notes
        -----
        The model assumes:
        - Uniform priors for calibrated parameters (bounds given by `parameters_range`)
        - Gaussian likelihood with no correlation between outputs
        All non-calibrated parameters are set to a constant value. This is chosen as the
        midpoint value of `parameters_range`.
        """
        TorchDeviceMixin.__init__(self, device=device)
        self.emulator = emulator
        # TODO: what if have tensor?
        # maybe always save this as a tensor?
        # BUT probably want to store the names somewhere (same as SA)
        self.parameter_range = parameter_range
        if calibration_params is None:
            calibration_params = list(parameter_range.keys())
        self.calibration_params = calibration_params
        self._process_observations(observations)
        self._process_obs_noise(list(observations.keys()), observation_noise)

    def _process_observations(
        self,
        observations: dict[str, float] | dict[str, list[float]],
    ):
        """
        Turn `observations` into tensor shaped [n_samples, n_ouputs], save attribute.

        Parameters
        ----------
        observations: dict[str, float] | dict[str, list[float]]
            A dictionary of either a single value or a list of values per output.
        """

        observation_values = [
            (torch.tensor(value) if isinstance(value, list) else torch.tensor([value]))
            for value in observations.values()
        ]
        # shape: [n_samples, n_outputs]
        self.observations = torch.stack(observation_values, dim=1).to(self.device)

    def _process_obs_noise(
        self,
        output_names: list[str],
        observation_noise: float | dict[str, float] = 0.1,
    ):
        """
        Ensure that `observation_noise` is handled correctly and saved as attribute:
        - if float, set the same observation noise value for all outputs
        - if dict, make sure the order matches `self.observations`

        Parameters
        ----------
        output_names: list[str]
            Names of output parameters in order of `self.observations`.
        observation_noise: float | dict[str, float]
           A single value or a dictionary of values (one per output). Defaults to 0.1.
        """
        if isinstance(observation_noise, float):
            # Broadcast to match outputs
            self.obs_noise = torch.full(
                (self.observations.shape[1],), observation_noise
            ).to(self.device)
        elif isinstance(observation_noise, dict):
            # Ensure order matches self.observations
            noise_values = [
                torch.tensor(observation_noise[key]) for key in output_names
            ]
            self.obs_noise = torch.tensor(noise_values).to(self.device)
        else:
            msg = "`observation_noise` must be a float or dict."
            raise ValueError(msg)

    def model(self):
        """Pyro model."""

        # Sample from uniform priors for calibration parameters
        calibration_params = {}
        for param in self.calibration_params:
            min_val, max_val = self.parameter_range[param]
            calibration_params[param] = pyro.sample(
                param, dist.Uniform(min_val, max_val)
            )

        # Set all other parameters to midpoint value in range
        # Ensure that full_params is shape [1, n_inputs]
        full_params = torch.zeros((1, len(self.parameter_range)), device=self.device)
        for i, param in enumerate(self.parameter_range.keys()):
            if param in calibration_params:
                full_params[0, i] = calibration_params[param]
            else:
                min_val, max_val = self.parameter_range[param]
                full_params[0, i] = (min_val + max_val) / 2

        # Emulator prediction
        with torch.no_grad():
            output = self.emulator.predict(full_params)
            if isinstance(output, TensorLike):
                y_pred = output
            elif isinstance(output, DistributionLike):
                y_pred = output.mean

        # Diagonal covariance (uncorrelated outputs)
        pred_cov = torch.diag(self.obs_noise.to(self.device))

        # Likelihood
        for i in range(self.observations.shape[0]):
            pyro.sample(
                f"obs_{i}",
                dist.MultivariateNormal(y_pred, covariance_matrix=pred_cov),
                obs=self.observations[i],
            )

    def run_mcmc(
        self, warmup_steps: int = 500, num_samples: int = 1000, num_chains: int = 1
    ) -> MCMC:
        """
        Run MCMC sampling with NUTS sampler.

        Parameters
        ----------
        warmup_steps: int
            Number warm up steps to run per chain (i.e., burn-in). These samples are
            discarded. Defaults to 500.
        num_samples: int
            Number of samples to draw after warm up. Defaults to 1000.
        num_chains: int
            Number of parallel chains to run. Defaults to 1.

        Returns
        -------
        MCMC
            The Pyro MCMC object. Use either `mcmc.summary()` or `mcmc.get_samples()`.
        """

        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(
            nuts_kernel,
            warmup_steps=warmup_steps,
            num_samples=num_samples,
            num_chains=num_chains,
            initial_params=self._set_initial_values(num_chains),
        )
        mcmc.run()
        return mcmc

    def predict(self, test_x: TensorLike) -> TensorLike:
        """
        Return posterior predictive.

        Parameters
        ----------
        test_x: TensorLike
            Tensor of parameters to make predictions for [n_data_samples, n_inputs].

        Returns
        -------
        TensorLike
            Tensor of posterior predictive predictions [n_mcmc_samples, n_outputs].
        """
        # TODO: check return shape, should we just do this for one data point?
        # TODO: imp;le
        return test_x

    def _set_initial_values(self, num_chains: int) -> None | dict[str, TensorLike]:
        """
        Set the initian parameter values for each MCMC chain.

        Parameters
        ----------
        num_chains: int
            Number of parallel chains to run. Defaults to 1.

        Returns
        -------
        None | dict[str, TensorLike]
        """
        # TODO
        # Dict containing initial tensors in unconstrained space to initiate the
        # markov chain. The leading dimension size must match that of num_chains.
        # If not specified, parameter values will be sampled from the prior.
        return
