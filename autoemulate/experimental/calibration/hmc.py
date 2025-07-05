import arviz as az
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS, Predictive

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
            A dictionary mapping input parameter names to their (min, max) ranges.
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
        self.parameter_range = parameter_range
        if calibration_params is None:
            calibration_params = list(parameter_range.keys())
        self.calibration_params = calibration_params
        self.emulator = emulator
        self.output_names = list(observations.keys())
        self._process_observations(observations)
        self._process_obs_noise(observation_noise)

    def _process_observations(
        self,
        observations: dict[str, float] | dict[str, list[float]],
    ):
        """
        Turn `observations` into tensor shaped [n_obs, n_ouputs], save attribute.

        Parameters
        ----------
        observations: dict[str, float] | dict[str, list[float]]
            A dictionary of either a single value or a list of values per output.
        """

        observation_values = [
            (torch.tensor(value) if isinstance(value, list) else torch.tensor([value]))
            for value in observations.values()
        ]
        # shape: [n_obs, n_outputs]
        self.observations = torch.stack(observation_values, dim=1).to(self.device)

    def _process_obs_noise(
        self,
        observation_noise: float | dict[str, float] = 0.1,
    ):
        """
        Convert `observation_noise` to tensor and save as attribute.

        Parameters
        ----------
        observation_noise: float | dict[str, float]
           A single value or a dictionary of values (one per output). Defaults to 0.1.
        """
        if isinstance(observation_noise, float):
            # Broadcast to match output size
            self.obs_noise = torch.full(
                (self.observations.shape[1],), observation_noise
            ).to(self.device)
        elif isinstance(observation_noise, dict):
            # Ensure order matches self.observations
            noise_values = [
                torch.tensor(observation_noise[key]) for key in self.output_names
            ]
            self.obs_noise = torch.tensor(noise_values).to(self.device)
        else:
            msg = "`observation_noise` must be a float or dict."
            raise ValueError(msg)

    def model(self, predict: bool = False):
        """
        Pyro model.

        Parameters
        ----------
        predict: bool
            Once MCMC has been run, one can call this methods to generate posterior
            predictive samples. Defaults to False.
        """

        # Set all input parameters, shape [1, n_inputs]
        full_params = torch.zeros((1, len(self.parameter_range)), device=self.device)

        # Each param is either sampled (if calibrated) or set to a constant value
        for i, param in enumerate(self.parameter_range.keys()):
            if param in self.calibration_params:
                # Set uniform priors for calibration parameters
                min_val, max_val = self.parameter_range[param]
                sampled_val = pyro.sample(param, dist.Uniform(min_val, max_val))
                full_params[0, i] = sampled_val
            else:
                # Set all other parameters to midpoint value in range
                min_val, max_val = self.parameter_range[param]
                full_params[0, i] = (min_val + max_val) / 2

        # Emulator prediction
        output = self.emulator.predict(full_params)
        if isinstance(output, TensorLike):
            pred_mean = output
        elif isinstance(output, DistributionLike):
            pred_mean = output.mean
        else:
            msg = "The emulator did not return a tensor or a distribution object."
            raise ValueError(msg)

        # MultivariateNormal likelihood over observations (diagonal covariance)
        for i, output in enumerate(self.output_names):
            pred_cov = self.obs_noise[i] * torch.eye(self.observations.shape[0]).to(
                self.device
            )
            if not predict:
                pyro.sample(
                    output,
                    dist.MultivariateNormal(
                        pred_mean[:, i], covariance_matrix=pred_cov
                    ),
                    obs=self.observations[:, i],
                )
            else:
                pyro.sample(
                    output,
                    dist.MultivariateNormal(
                        pred_mean[:, i], covariance_matrix=pred_cov
                    ),
                )

    def run_mcmc(
        self,
        warmup_steps: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        initial_params: dict[str, TensorLike] | None = None,
    ) -> MCMC:
        """
        Run MCMC sampling with NUTS sampler.

        Parameters
        ----------
        warmup_steps: int
            Number of warm up steps to run per chain (i.e., burn-in). These samples are
            discarded. Defaults to 500.
        num_samples: int
            Number of samples to draw after warm up. Defaults to 1000.
        num_chains: int
            Number of parallel chains to run. Defaults to 1.
        initial_params: dict[str, TensorLike] | None
            Optional dictionary specifiying initial values for each calibration
            parameters. The list length must be the same as the number of chains.

        Returns
        -------
        MCMC
            The Pyro MCMC object, methods include `summary()` and `get_samples()`.
        """

        # TODO: add checks for initial params
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(
            nuts_kernel,
            warmup_steps=warmup_steps,
            num_samples=num_samples,
            num_chains=num_chains,
            # If None, parameter init values for each chain are sampled from the prior.
            initial_params=initial_params,
        )
        mcmc.run()
        return mcmc

    def posterior_predictive(self, mcmc: MCMC) -> TensorLike:
        """
        Return posterior predictive samples.

        Parameters
        ----------
        mcmc: MCMC
            The MCMC object.

        Returns
        -------
        TensorLike
            Tensor of posterior predictive samples [n_mcmc_samples, n_obs, n_outputs].
        """
        posterior_samples = mcmc.get_samples()
        posterior_predictive = Predictive(self.model, posterior_samples)
        return posterior_predictive(predict=True)

    def to_arviz(
        self, mcmc: MCMC, posterior_predictive: bool = False
    ) -> az.data.inference_data.InferenceData:
        """
        Convert MCMC object to Arviz InferenceData object to enable plotting.

        Parameters
        ----------
        mcmc: MCMC
            The MCMC object.
        posterior_predictive: bool
            Whether to include posterior predictive samples.

        Returns
        -------
        az.data.inference_data.InferenceData
        """
        pp_samples = None
        if posterior_predictive:
            pp_samples = self.posterior_predictive(mcmc)
        return az.from_pyro(mcmc, posterior_predictive=pp_samples)
