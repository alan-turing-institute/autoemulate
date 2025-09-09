import pyro
import pyro.distributions as dist
import torch

from autoemulate.calibration.base import BayesianMixin
from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.logging_config import get_configured_logger
from autoemulate.core.types import DeviceLike, TensorLike
from autoemulate.emulators.base import Emulator


class BayesianCalibration(TorchDeviceMixin, BayesianMixin):
    """
    Bayesian calibration using Markov Chain Monte Carlo (MCMC).

    Bayesian calibration estimates the probability distribution over input parameters
    given observed data, providing uncertainty estimates.
    """

    def __init__(
        self,
        emulator: Emulator,
        parameter_range: dict[str, tuple[float, float]],
        observations: dict[str, TensorLike],
        observation_noise: float | dict[str, float] = 0.01,
        model_uncertainty: bool = False,
        model_discrepancy: float = 0.0,
        calibration_params: list[str] | None = None,
        device: DeviceLike | None = None,
        log_level: str = "progress_bar",
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
            A single value or a dictionary of values (one per output) of measurement
            noise measured in terms of variance. Defaults to 0.01.
        model_uncertainty: bool
            Whether to include the variance associated with model predictions when
            calculating the likelihood. Defaults to False.
        model_discrepancy:
            Additional uncertainty to include in the likelihood, specified as a
            variance. This is equivalent to the discrepancy term used in history
            matching to represent uncertainty about model validity. Defaults to 0.0.
        calibration_params: list[str] | None
            Optional list of input parameters to calibrate. Any parameters that are not
            listed will be set to the midpoint value of their parameter range. If None,
            will calibrate all input parameters. Defaults to None.
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.
            TODO: do we need to do anything more to ensure the device is correctly
            handled for the pyro model?
        log_level: str
            Logging level for the calibration. Can be one of:
            - "progress_bar": shows a progress bar during batch simulations
            - "debug": shows debug messages
            - "info": shows informational messages
            - "warning": shows warning messages
            - "error": shows error messages
            - "critical": shows critical messages

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
        self.logger, self.progress_bar = get_configured_logger(log_level)
        self.logger.info(
            "Initializing BayesianCalibration with parameters: %s",
            self.calibration_params,
        )

        # Check observation tensors are 1D (convert if 0D)
        processed_observations = {}
        obs_lengths = []
        for output, obs in observations.items():
            if obs.ndim == 0:
                corrected_obs = obs.unsqueeze(0)
                self.logger.debug(
                    "Observation for output '%s' converted from 0D to 1D.",
                    output,
                )
            elif obs.ndim > 1:
                self.logger.error("Tensor for output '%s' is not 1D.", output)
                raise ValueError(f"Tensor for output '{output}' is not 1D.")
            else:
                corrected_obs = obs
            processed_observations[output] = corrected_obs.to(self.device)
            obs_lengths.append(corrected_obs.shape[0])
        if len(set(obs_lengths)) != 1:
            msg = "All outputs must have the same number of observations."
            self.logger.error(msg)
            raise ValueError(msg)
        self.observations = processed_observations
        self.n_observations = obs_lengths[0]
        self.logger.info("Processed observations for outputs: %s", self.output_names)

        # Save observation noise as {output: value} dictionary
        if isinstance(observation_noise, float):
            self.observation_noise = dict.fromkeys(
                self.output_names,
                torch.tensor(observation_noise).to(self.device),
            )
            self.logger.debug(
                "Observation noise (variance) set as float: %s", observation_noise
            )
        elif isinstance(observation_noise, dict):
            self.observation_noise = observation_noise
            self.logger.debug(
                "Observation noise (variance) set as dict: %s", observation_noise
            )
        else:
            msg = (
                "Observation noise (variance) must be either a float or a dictionary "
                "of floats."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        self.model_uncertainty = model_uncertainty
        self.model_discrepancy = model_discrepancy

    def model(self, predict: bool = False):
        """
        Pyro model.

        Parameters
        ----------
        predict: bool
            Whether to run the model with existing samples to generate posterior
            predictive distribution. Used with `pyro.infer.Predictive`.
        """
        # Pre-allocate tensor for all input parameters, shape [1, n_inputs]
        param_list = []
        # Each param is either sampled (if calibrated) or set to a constant value
        for param in self.parameter_range:
            if param in self.calibration_params:
                # Sample from uniform prior
                min_val, max_val = self.parameter_range[param]
                sampled_val = pyro.sample(param, dist.Uniform(min_val, max_val))
                param_list.append(sampled_val.to(self.device))
            else:
                # Set to midpoint value in parameter range
                min_val, max_val = self.parameter_range[param]
                midpoint_val = (min_val + max_val) / 2
                param_list.append(torch.tensor(midpoint_val, device=self.device))
        full_params = torch.stack(param_list, dim=0).unsqueeze(0).float()

        # Get emulator prediction
        mean, variance = self.emulator.predict_mean_and_variance(
            full_params, with_grad=True
        )

        # Likelihood
        for i, output in enumerate(self.output_names):
            # Create combined scale (stddev) for Normal from the following variances:
            # - observation noise
            # - model discrepancy
            # - model uncertainty (if specified and provided by emulator)

            # Get observation noise
            observation_variance = self.observation_noise[output]

            # Combine variances (add prediction variance if required and available)
            total_variance = (
                observation_variance + self.model_discrepancy + variance[0, i]
                if self.model_uncertainty and variance is not None
                else torch.tensor(observation_variance + self.model_discrepancy).to(
                    self.device
                )
            )
            # Take sqrt for final scale (stddev)
            scale = total_variance.sqrt()

            if not predict:
                with pyro.plate(f"data_{output}", self.n_observations):
                    assert self.observations is not None
                    pyro.sample(
                        output,
                        dist.Normal(mean[0, i], scale),
                        obs=self.observations[output],
                    )
            else:
                with pyro.plate(f"data_{output}", self.n_observations):
                    pyro.sample(
                        output,
                        dist.Normal(mean[0, i], scale),
                    )
