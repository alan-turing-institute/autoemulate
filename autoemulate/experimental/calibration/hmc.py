import arviz as az
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import HMC, MCMC, NUTS, Predictive
from pyro.infer.mcmc import RandomWalkKernel

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.types import DeviceLike, DistributionLike, TensorLike


class MCMC_calibration(TorchDeviceMixin):
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
        parameter_range: dict[str, tuple[float, float]],
        observations: dict[str, TensorLike],
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
        observations: dict[str, TensorLike]
            A dictionary of observations for each output.
        observation_noise: float |  dict[str, float]
            A single value or a dictionary of values (one per output). Defaults to 0.1.
        calibration_params: list[str] | None
            Optional list of input parameters to calibrate. Any parameters that are not
            listed will be set to the midpoint value of their parameter range. If None,
            will calibrate all input parameters. Defaults to None.
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.
            TODO: do we need to do anything more to ensure the device is correctly
            handled for the pyro model?

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
        self.emulator.device = self.device
        self.output_names = list(observations.keys())

        # Check observation tensors are 1D (convert if 0D)
        processed_observations = {}
        obs_lengths = []
        for output, obs in observations.items():
            if obs.ndim == 0:
                corrected_obs = obs.unsqueeze(0)
            elif obs.ndim > 1:
                raise ValueError(f"Tensor for output '{output}' is not 1D.")
            else:
                corrected_obs = obs
            processed_observations[output] = corrected_obs.to(self.device)
            obs_lengths.append(corrected_obs.shape[0])
        if len(set(obs_lengths)) != 1:
            msg = "All outputs must have the same number of observations."
            raise ValueError(msg)
        self.observations = processed_observations
        self.n_observations = obs_lengths[0]

        # Save observation noise as {output: value} dictionary
        if isinstance(observation_noise, float):
            self.observation_noise = dict.fromkeys(self.output_names, observation_noise)
        elif isinstance(observation_noise, dict):
            self.observation_noise = observation_noise
        else:
            msg = "Noise must be either a float or a dictionary of floats."
            raise ValueError(msg)

    def _get_kernel(self, sampler: str, **sampler_kwargs):
        """Get the appropriate MCMC kernel based on sampler choice."""
        sampler = sampler.lower()

        if sampler == "nuts":
            return NUTS(self.model, **sampler_kwargs)
        if sampler == "hmc":
            step_size = sampler_kwargs.pop("step_size", 0.01)
            trajectory_length = sampler_kwargs.pop("trajectory_length", 1.0)
            return HMC(
                self.model,
                step_size=step_size,
                trajectory_length=trajectory_length,
                **sampler_kwargs,
            )
        if sampler == "metropolis":
            return RandomWalkKernel(self.model, **sampler_kwargs)
        raise ValueError(f"Unknown sampler: {sampler}")

    def model(self, predict: bool = False):
        """
        Pyro model.

        Parameters
        ----------
        predict: bool
            Indicates whether to sample without conditioning on data. This is used to
            generate posterior predictive samples. Defaults to False.
        """

        # Pre-allocate tensor for all input parameters, shape [1, n_inputs]
        full_params = torch.zeros((1, len(self.parameter_range)), device=self.device)

        # Each param is either sampled (if calibrated) or set to a constant value
        for i, param in enumerate(self.parameter_range.keys()):
            if param in self.calibration_params:
                # Sample from uniform prior
                min_val, max_val = self.parameter_range[param]
                sampled_val = pyro.sample(param, dist.Uniform(min_val, max_val))
                full_params[0, i] = sampled_val.to(self.device)

            else:
                # Set to midpoint value in parameter range
                min_val, max_val = self.parameter_range[param]
                full_params[0, i] = torch.tensor(
                    (min_val + max_val) / 2, device=self.device
                )

        # Get emulator prediction
        output = self.emulator.predict(full_params)
        if isinstance(output, TensorLike):
            pred_mean = output.to(self.device)

        elif isinstance(output, DistributionLike):
            pred_mean = output.mean.to(self.device)
        else:
            msg = "The emulator did not return a tensor or a distribution object."
            raise ValueError(msg)

        # Use MultivariateNormal likelihood to handle multiple observations
        diag = torch.eye(self.n_observations).to(self.device)
        for i, output in enumerate(self.output_names):
            mvn_mean = pred_mean[:, i].expand(self.n_observations)
            mvn_cov = self.observation_noise[output] * diag
            if not predict:
                pyro.sample(
                    output,
                    dist.MultivariateNormal(mvn_mean, covariance_matrix=mvn_cov),
                    obs=self.observations[output],
                )
            else:
                pyro.sample(
                    output,
                    dist.MultivariateNormal(mvn_mean, covariance_matrix=mvn_cov),
                )

    def run(
        self,
        warmup_steps: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        initial_params: dict[str, TensorLike] | None = None,
        sampler: str = "nuts",
        **sampler_kwargs,
    ) -> MCMC:
        """
        Run MCMC with NUTS sampler.

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
            parameter per chain. The tensors must be of length `num_chains`.

        Returns
        -------
        MCMC
            The Pyro MCMC object. Methods include `summary()` and `get_samples()`.
        """

        # Check initial param values match number of chains

        if initial_params is not None:
            for param, init_vals in initial_params.items():
                if init_vals.shape[0] != num_chains:
                    msg = (
                        "An initial value must be provided for each chain, parameter "
                        f"{param} tensor only has {init_vals.shape[0]} values."
                    )
                    raise ValueError(msg)

        # Run NUTS
        kernel = self._get_kernel(sampler, **sampler_kwargs)
        mcmc = MCMC(
            kernel,
            warmup_steps=warmup_steps,
            num_samples=num_samples,
            num_chains=num_chains,
            # If None, parameter init values for each chain
            # are sampled from the prior.
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
        Convert MCMC object to Arviz InferenceData object for plotting.

        Parameters
        ----------
        mcmc: MCMC
            The MCMC object.
        posterior_predictive: bool
            Whether to include posterior predictive samples. Defaults to False.

        Returns
        -------
        az.data.inference_data.InferenceData
        """
        pp_samples = None
        if posterior_predictive:
            pp_samples = self.posterior_predictive(mcmc)

        # need to create dataset manually for Metropolis Hastings
        # this is because az.from_pyro expects kernal with `divergences`
        if isinstance(mcmc.kernel, RandomWalkKernel):
            if posterior_predictive:
                az_data = az.InferenceData(
                    posterior=az.convert_to_dataset(mcmc.get_samples()),
                    posterior_predictive=az.convert_to_dataset(pp_samples),
                    observed_data=az.convert_to_dataset(self.observations),
                )
            else:
                az_data = az.InferenceData(
                    posterior=az.convert_to_dataset(mcmc.get_samples()),
                )
        else:
            az_data = az.from_pyro(mcmc, posterior_predictive=pp_samples)

        return az_data
