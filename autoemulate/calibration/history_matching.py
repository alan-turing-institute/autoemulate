import logging
import warnings

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.results import Result
from autoemulate.core.types import DeviceLike, DistributionLike, TensorLike
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators import TransformedEmulator, get_emulator_class
from autoemulate.simulations.base import Simulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autoemulate")


class HistoryMatching(TorchDeviceMixin):
    r"""
    History Matching class for model calibration.

    History matching is a model calibration method, which uses observed data to
    rule out ``implausible`` parameter values. The implausibility metric is:

    .. math::

        I_i(\bar{x_0}) = \frac{|z_i - \mathbb{E}(f_i(\bar{x_0}))|}
        {\sqrt{\text{Var}[z_i - \mathbb{E}(f_i(\bar{x_0}))]}}

    Queried parameters above a given implausibility threshold are ruled out (RO)
    whereas all other parameters are marked as not ruled out yet (NROY).
    """

    def __init__(
        self,
        observations: dict[str, tuple[float, float]] | dict[str, float],
        threshold: float = 3.0,
        model_discrepancy: float = 0.0,
        rank: int = 1,
        device: DeviceLike | None = None,
    ):
        """
        Initialize the history matching object.

        Parameters
        ----------
        observations: dict[str, tuple[float, float] | dict[str, float]
            For each output variable, specifies observed [value, noise]. In case
            of no uncertainty in observations, provides just the observed value.
        threshold: float
            Implausibility threshold (query points with implausibility scores that
            exceed this value are ruled out). Defaults to 3, which is considered
            a good value for simulations with a single output.
        model_discrepancy: float
            Additional variance to include in the implausibility calculation.
        rank: int
            Scoring method for multi-output problems. Must be 1 <= rank <= n_outputs.
            When the implausibility scores are ordered across outputs, it indicates
            which rank to use when determining whether the query point is NROY. The
            default of ``1`` indicates that the largest implausibility will be used.
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.
        """
        TorchDeviceMixin.__init__(self, device=device)

        self.threshold = threshold
        self.discrepancy = model_discrepancy
        self.out_dim = len(observations)

        if rank > self.out_dim or rank < 1:
            raise ValueError(
                f"Rank ({rank}) is outside valid range between 1 and output dimension "
                f"of simulator ({self.out_dim})",
            )
        self.rank = rank

        # Save mean and variance of observations, shape: [1, n_outputs]
        self.obs_means, self.obs_vars = self._process_observations(observations)

    def _process_observations(
        self,
        observations: dict[str, tuple[float, float]] | dict[str, float],
    ) -> tuple[TensorLike, TensorLike]:
        """
        Turn observations into tensors of shape [1, n_inputs].

        Parameters
        ----------
        observations: dict[str, tuple[float, float] | dict[str, float]
            For each output variable, specifies observed [value, noise]. In case
            of no uncertainty in observations, provides just the observed value.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            Tensors of observations and the associated noise (which can be 0).
        """
        values = torch.tensor(list(observations.values()), device=self.device)

        # No variance
        if values.ndim == 1:
            means = values
            variances = torch.zeros_like(means, device=self.device)
        # Values are (mean, variance)
        elif values.ndim == 2:
            means = values[:, 0]
            variances = values[:, 1]
        else:
            msg = "Observations must be either float or tuple of two floats."
            raise ValueError(msg)

        # Reshape observation tensors for broadcasting
        return means.view(1, -1), variances.view(1, -1)

    def _create_nroy_mask(self, implausibility: TensorLike) -> TensorLike:
        """
        Create mask for NROY points based on rank.

        Parameters
        ----------
        implausibility: TensorLike
            Tensor of implausibility scores for tested parameters.

        Returns
        -------
        TensorLike
            Tensor indicating whether each implausability score is NROY
            given self.rank and self.threshold values.
        """
        # Sort implausibilities for each sample (descending)
        I_sorted, _ = torch.sort(implausibility, dim=1, descending=True)
        # The rank-th highest output implausibility must be <= threshold
        return I_sorted[:, self.rank - 1] <= self.threshold

    def get_nroy(
        self, implausibility: TensorLike, x: TensorLike | None = None
    ) -> TensorLike:
        """
        Get indices of NROY points from implausibility scores.

        If `x` is provided, returns parameter values at NROY indices.

        Parameters
        ----------
        implausibility: TensorLike
            Tensor of implausibility scores for tested input parameters.
        x: Tensorlike | None
            Optional tensor of scored input parameters.

        Returns
        -------
        TensorLike
            Indices of NROY points or `x` parameters at NROY indices.
        """
        nroy_mask = self._create_nroy_mask(implausibility)
        idx = torch.where(nroy_mask)[0]
        if x is None:
            return idx
        return x[idx]

    def get_ro(
        self, implausibility: TensorLike, x: TensorLike | None = None
    ) -> TensorLike:
        """
        Get indices of RO points from implausibility scores.

        If `x` is provided, returns parameter values at RO indices.

        Parameters
        ----------
        implausibility: TensorLike
            Tensor of implausibility scores for tested input parameters.
        x: Tensorlike | None
            Optional tensor of scored iput parameters.

        Returns
        -------
        TensorLike
            Indices of RO points or `x` parameters at RO indices.
        """
        nroy_mask = self._create_nroy_mask(implausibility)
        idx = torch.where(~nroy_mask)[0]
        if x is None:
            return idx
        return x[idx]

    def calculate_implausibility(
        self,
        pred_means: TensorLike,  # [n_samples, n_outputs]
        pred_vars: TensorLike,  # [n_samples, n_outputs]
    ) -> TensorLike:
        """
        Calculate implausibility scores.

        Parameters
        ----------
        pred_means: TensorLike
            Tensor of prediction means [n_samples, n_outputs]
        pred_vars: TensorLike
            Tensor of prediction variances [n_samples, n_outputs].

        Returns
        -------
        TensorLike
            Tensor of implausibility scores.
        """
        # Additional variance due to model discrepancy (defaults to 0)
        discrepancy = torch.full_like(
            self.obs_vars, self.discrepancy, device=self.device
        )

        # Calculate total variance
        Vs = pred_vars + discrepancy + self.obs_vars

        # Calculate implausibility
        return torch.abs(self.obs_means - pred_means) / torch.sqrt(Vs)

    @staticmethod
    def generate_param_bounds(
        nroy_x: TensorLike,
        buffer_ratio: float = 0.05,
        param_names: list[str] | None = None,
        min_samples: int = 1,
    ) -> dict[str, tuple[float, float]] | None:
        """
        Generate lower/upper parameter bounds as min/max of NROY samples.

        Parameters
        ----------
        nroy_x: TensorLike
            A tensor of NROY parameter samples [n_samples, n_inputs].
        buffer_ratio: float
            A scaling factor used to expand the bounds of the (NROY) parameter space.
            It is applied as a ratio of the range (max_val - min_val) of each input
            parameter to create a buffer around the NROY minimum and maximum values.
        param_names: list[str] | None
            Optional list of parameter names. If None, uses default `["x1", ..., "xn"]`.
        min_samples: int
            Minimum number of samples needed to generate new bounds.

        Returns
        -------
        dict[str, [float, float]] | None
            The generated [lower, upper] parameter bounds. Returns None if there are
            not enough samples to generate bounds from.
        """
        if param_names is None:
            param_names = [f"x{i + 1}" for i in range(nroy_x.shape[1])]

        if nroy_x.shape[0] > min_samples:
            min_val = torch.min(nroy_x, dim=0).values
            max_val = torch.max(nroy_x, dim=0).values
            buffer = (max_val - min_val) * buffer_ratio
            lower_bound = min_val - buffer
            upper_bound = max_val + buffer

            return {
                param: (lower_bound[i].item(), upper_bound[i].item())
                for i, param in enumerate(param_names)
            }
        return None


class HistoryMatchingWorkflow(HistoryMatching):
    """
    History Matching Workflow class.

    Run history matching workflow:
    - sample parameter values to test from the current NROY parameter space
    - use emulator to rule out implausible parameters and update NROY space
    - make predictions for a subset the NROY parameters using the simulator
    - refit the emulator using the simulated data
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        simulator: Simulator,
        result: Result,
        observations: dict[str, tuple[float, float]] | dict[str, float],
        threshold: float = 3.0,
        model_discrepancy: float = 0.0,
        rank: int = 1,
        train_x: TensorLike | None = None,
        train_y: TensorLike | None = None,
        device: DeviceLike | None = None,
        random_seed: int | None = None,
    ):
        """
        Initialize the history matching workflow object.

        Parameters
        ----------
        simulator: Simulator
            A simulator.
        result: Result
            A Result object containing the pre-trained emulator and its parameters.
        observations: dict[str, tuple[float, float] | dict[str, float]
            For each output variable, specifies observed [value, noise]. In case
            of no uncertainty in observations, provides just the observed value.
        threshold: float
            Implausibility threshold (query points with implausibility scores that
            exceed this value are ruled out). Defaults to 3, which is considered
            a good value for simulations with a single output.
        model_discrepancy: float
            Additional variance to include in the implausibility calculation.
        rank: int
            Scoring method for multi-output problems. Must be 1 <= rank <= n_outputs.
            When the implausibility scores are ordered across outputs, it indicates
            which rank to use when determining whether the query point is NROY. The
            default val of ``1`` indicates that the largest implausibility will be used.
        train_x: TensorLike | None
            Optional tensor of input data the emulator was trained on.
        train_y: TensorLike | None
            Optional tensor of output data the emulator was trained on.
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.
        """
        super().__init__(observations, threshold, model_discrepancy, rank, device)
        self.simulator = simulator
        if random_seed is not None:
            set_random_seed(seed=random_seed)
        self.result = result
        self.emulator = result.model
        self.emulator.device = self.device

        # New data is generated in `run()` and appended here
        # The emulator is then refit on all data collected so far
        if train_x is not None and train_y is not None:
            self.train_x = train_x.float().to(self.device)
            self.train_y = train_y.float().to(self.device)
        else:
            self.train_x = torch.empty((0, self.simulator.in_dim), device=self.device)
            self.train_y = torch.empty((0, self.simulator.out_dim), device=self.device)

        # NROY samples are also generated in `run()` and used in `cloud_sample()`
        # We only ever use the most recent NROY samples
        # This means `self.nroy_samples` gets overwritten each time `run()` is called
        self.nroy_samples = None

    def _is_within_bounds(
        self, sample: TensorLike, bounds_dict: dict[str, tuple[float, float]]
    ) -> bool:
        """
        Check if `sample` is within the bounds defined in `bounds_dict`.

        Parameters
        ----------
        sample: torch.Tensor
            A single sample of input parameters to check, shape [1, in_dim].
        bounds_dict: dict of {param_name: [lower, upper]}
            A dictionary of parameter bounds for each parameter.

        Returns
        -------
        bool
            True if the sample is within the bounds, False otherwise.
        """
        sample = sample.squeeze(0)  # shape: [in_dim]
        lowers = torch.tensor(
            [bounds[0] for bounds in bounds_dict.values()],
            dtype=sample.dtype,
            device=sample.device,
        )
        uppers = torch.tensor(
            [bounds[1] for bounds in bounds_dict.values()],
            dtype=sample.dtype,
            device=sample.device,
        )
        return bool(torch.all((sample >= lowers) & (sample <= uppers)).item())

    def _sample_within_bounds(
        self,
        dist: DistributionLike,
        bounds: dict[str, tuple[float, float]],
        n: int,
        constant_params: dict[int, float] | None = None,
        sample_params_idx: list[int] | None = None,
    ) -> list[TensorLike]:
        """
        Sample from distribution until `n` valid samples within the bounds are obtained.

        Handles constant parameters by inserting their values at the correct indices.

        Parameters
        ----------
        dist: DistributionLike
            A distribution to sample from, e.g., MultivariateNormal.
        bounds: dict[str, tuple[float, float]]
            A dictionary of [min, max] parameter bounds for each sampled parameter.
        n: int
            The number of samples to generate.
        constant_params: dict[int, float] | None
            A dictionary of constant parameter indices and their values.
        sample_params_idx: list[int]
            Indices of parameters that are not constant.

        Returns
        -------
        list[TensorLike]
            A list of valid samples that are within the bounds.
        """
        param_dim = len(bounds) + (len(constant_params) if constant_params else 0)
        if sample_params_idx is None:
            sample_params_idx = list(range(len(bounds)))

        valid_samples = []
        while len(valid_samples) < n:
            n_remaining = n - len(valid_samples)
            samples = dist.sample((n_remaining,))
            full = torch.empty(
                (n_remaining, param_dim), dtype=samples.dtype, device=samples.device
            )
            if constant_params:
                const_idx = list(constant_params.keys())
                const_vals = torch.tensor(
                    list(constant_params.values()),
                    dtype=samples.dtype,
                    device=samples.device,
                )
                full[:, const_idx] = const_vals
            full[:, sample_params_idx] = samples
            valid_samples.extend([s for s in full if self._is_within_bounds(s, bounds)])
        return valid_samples

    def cloud_sample(self, n: int, scaling_factor: float = 0.1) -> TensorLike:
        """
        Generate `n` additional parameter samples using cloud sampling.

        Handles fixed parameters (min == max) by not sampling those and inserting
        their constant values.

        Parameters
        ----------
        n: int
            The number of samples to generate.
        scaling_factor: float
            The standard deviation of the Gaussian to sample from in cloud sampling is
            set to: `parameter range * scaling_factor`.

        Returns
        -------
        TensorLike
            A tensor of sampled parameters of shape [n, in_dim].
        """
        assert isinstance(self.nroy_samples, TensorLike)

        bounds = self.generate_param_bounds(self.nroy_samples, buffer_ratio=0.0)
        assert bounds is not None

        # Identify constant parameters
        min_vals = torch.tensor([b[0] for b in bounds.values()], device=self.device)
        max_vals = torch.tensor([b[1] for b in bounds.values()], device=self.device)
        is_constant = min_vals == max_vals
        constant_params = {
            i: min_vals[i].item() for i, fixed in enumerate(is_constant) if fixed
        }
        sample_params_idx = [i for i, fixed in enumerate(is_constant) if not fixed]

        # If all parameters are constant just return the constant sample n times
        if len(sample_params_idx) == 0:
            return min_vals.unsqueeze(0).repeat(n, 1)

        # Only use non-constant parameters for mean and covariance to sample from
        nroy_params_to_sample = self.nroy_samples[:, sample_params_idx]
        stdev = (
            nroy_params_to_sample.max(dim=0).values
            - nroy_params_to_sample.min(dim=0).values
        ) * scaling_factor
        covariance_matrix = torch.diag(stdev**2)

        # Shuffle the order of means to sample from
        num_means = nroy_params_to_sample.shape[0]
        perm = torch.randperm(num_means, device=nroy_params_to_sample.device)

        # Determine how many samples to draw for each mean, handle remainder
        min_samples_per_mean = n // num_means
        remainder_to_sample = n % num_means

        all_valid_samples = []
        for i, mean in enumerate(nroy_params_to_sample[perm]):
            n_samples = min_samples_per_mean + (1 if i < remainder_to_sample else 0)
            mvn = MultivariateNormal(mean, covariance_matrix)
            all_valid_samples.extend(
                self._sample_within_bounds(
                    mvn, bounds, n_samples, constant_params, sample_params_idx
                )
            )

        return torch.stack(all_valid_samples, dim=0)

    def generate_samples(
        self, n: int, scaling_factor: float = 0.1
    ) -> tuple[TensorLike, TensorLike]:
        """
        Generate parameter samples and evaluate implausibility.

        Draw `n` samples either from the simulator min/max parameter bounds or
        using cloud sampling centered at NROY samples. Evaluate sample
        implausability using emulator predictions.

        Parameters
        ----------
        n: int
            The number of parameter samples to generate.
        scaling_factor: float
            The standard deviation of the Gaussian used in cloud sampling is
            set to: `parameter range * scaling_factor`.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tensor of tested input parameters and their implausability scores.
        """
        # Generate `n` parameter samples (use simulator if have no NROY samples)
        if self.nroy_samples is None:
            test_x = self.simulator.sample_inputs(n).to(self.device)
        else:
            test_x = self.cloud_sample(n, scaling_factor).to(self.device)

        # Rule out implausible parameters from samples using an emulator
        mean, variance = self.emulator.predict_mean_and_variance(test_x)
        impl_scores = self.calculate_implausibility(mean, variance)

        return test_x, impl_scores

    def sample_tensor(self, n: int, x: TensorLike) -> TensorLike:
        """
        Randomly sample `n` rows from `x`.

        Parameters
        ----------
        n: int
            The number of samples to draw.
        x: TensorLike
            The tensor to sample from.

        Returns
        -------
        TensorLike
            A tensor of samples with `n` rows.
        """
        if x.shape[0] < n:
            warnings.warn(
                f"Number of tensor rows {x.shape[0]} is less than {n} samples.",
                stacklevel=2,
            )
        idx = torch.randperm(x.shape[0], device=self.device)[:n]
        return x[idx]

    def simulate(self, x: TensorLike) -> tuple[TensorLike, TensorLike]:
        """
        Simulate `x` parameter inputs and filter out failed simulations.

        Parameters
        ----------
        x: TensorLike
            A tensor of parameters to simulate [n_samples, n_inputs].

        Returns
        -------
        tuple[TensorLike, TensorLike]
            Tensors of succesfully simulated input parameters and predictions.
        """
        # if simulation fails, returned y and x have fewer rows than input x
        y, x = self.simulator.forward_batch_skip_failures(x)
        y = y.to(self.device)
        x = x.to(self.device)

        self.train_y = torch.cat([self.train_y, y], dim=0)
        self.train_x = torch.cat([self.train_x, x], dim=0)

        return x, y

    def refit_emulator(self):
        """Refit the emulator using all available training data."""
        # Create a fresh model with the same configuration
        self.emulator = TransformedEmulator(
            self.train_x,
            self.train_y,
            model=get_emulator_class(self.result.model_name),
            x_transforms=self.result.x_transforms,
            y_transforms=self.result.y_transforms,
            device=self.device,
            **self.result.params,
        )
        # Fit the fresh model on the new data
        logger.info(" Refitting emulator.")
        self.emulator.fit(self.train_x, self.train_y)

    def run(
        self,
        n_simulations: int = 100,
        n_test_samples: int = 10000,
        max_retries: int = 3,
        scaling_factor: float = 0.1,
    ) -> tuple[TensorLike, TensorLike]:
        """
        Run a wave of the history matching workflow.

        Parameters
        ----------
        n_simulations: int
            Number of simulations to run.
        n_test_samples: int
            Number of input parameters to test for implausibility with the emulator.
            Parameters to simulate are sampled from this NROY subset.
        max_retries: int
            Maximum number of times to try to generate `n_simulations` NROY parameters.
            That is the maximum number of times to repeat the following steps:
                - draw `n_test_samples` parameters (use cloud sampling if possible)
                - use emulator to make predictions for those parameters
                - score implausability of parameters given predictions
                - identify NROY parameters within this set
        scaling_factor: float
            The standard deviation of the Gaussian to sample from in cloud sampling is
            set to: `parameter range * scaling_factor`.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tensor of tested input parameters and their implausibility scores from
            which simulation samples were then selected.
        """
        logger.debug(
            " Running history matching workflow with"
            " %d simulations and %d test samples.",
            n_simulations,
            n_test_samples,
        )

        test_parameters_list, impl_scores_list, nroy_parameters_list = (
            [],
            [],
            [torch.empty((0, self.simulator.in_dim), device=self.device)],
        )

        retries = 0
        while torch.cat(nroy_parameters_list, 0).shape[0] < n_simulations:
            if retries == max_retries:
                msg = (
                    f"Could not generate n_simulations ({n_simulations}) samples "
                    f"that are NROY after {max_retries} retries. "
                    f"Only {torch.cat(nroy_parameters_list, 0).shape[0]} "
                    "samples generated."
                )
                raise Warning(msg)
                break

            # Generate `n_test_samples` with implausability scores, identify NROY
            test_parameters, impl_scores = self.generate_samples(
                n_test_samples, scaling_factor
            )
            nroy_parameters = self.get_nroy(impl_scores, test_parameters)

            # Store results
            nroy_parameters_list.append(nroy_parameters)
            test_parameters_list.append(test_parameters)
            impl_scores_list.append(impl_scores)

            retries += 1

        # Next time that call run(), will sample using these NROY points
        self.nroy_samples = torch.cat(nroy_parameters_list, 0)

        # Randomly pick at most `n_simulations` parameters from NROY to simulate
        nroy_simulation_samples = self.sample_tensor(n_simulations, self.nroy_samples)

        # Make predictions using simulator (this updates self.x_train and self.y_train)
        _, _ = self.simulate(nroy_simulation_samples)

        # Refit emulator using all available data
        self.refit_emulator()

        # prediction = self.emulator.predict(self.train_x)

        # print("mean", ((prediction.mean - (self.train_y)) / self.train_y).mean())
        # print("std", ((prediction.mean - self.train_y) / self.train_y).std())

        # print("ratio", (prediction.variance / self.train_y).mean())
        # print("ratio", (prediction.variance / self.train_y).std())

        # print(
        #     "pred variance mean", (prediction.variance / prediction.mean).mean()
        # )
        # print("pred variance std", (prediction.variance / prediction.mean).std())

        # Return test parameters and impl scores for this run/wave
        return torch.cat(test_parameters_list, 0), torch.cat(impl_scores_list, 0)

    def run_waves(  # noqa: PLR0913
        self,
        n_waves: int = 5,
        frac_nroy_stop: float = 0.9,
        n_simulations: int = 100,
        n_test_samples: int = 10000,
        max_retries: int = 3,
        scaling_factor: float = 0.1,
    ) -> list[tuple[TensorLike, TensorLike]]:
        """
        Run multiple waves of the history matching workflow.

        Parameters
        ----------
        n_waves: int
            The maximum number of waves to run.
        frac_nroy_stop: float
            Fraction of NROY samples to stop at. If less than this fraction of
            NROY samples is reached, the workflow stops.
        n_simulations: int
            Number of simulations to run in each wave.
        n_test_samples: int
            Number of input parameters to test for implausibility with the emulator.
            Parameters to simulate are sampled from this NROY subset.
        max_retries: int
            Maximum number of times to try to generate `n_simulations` NROY parameters.
            That is the maximum number of times to repeat the following steps:
                - draw `n_test_samples` parameters (use cloud sampling if possible)
                - use emulator to make predictions for those parameters
                - score implausibility of parameters given predictions
                - identify NROY parameters within this set
        scaling_factor: float
            The standard deviation of the Gaussian to sample from in cloud sampling is
            set to: `parameter range * scaling_factor`.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tensor of tested input parameters and their implausibility scores.
        """
        wave_results = []
        for i in range(n_waves):
            logger.info(" Running history matching wave %d/%d", i + 1, n_waves)
            test_x, impl_scores = self.run(
                n_simulations=n_simulations,
                n_test_samples=n_test_samples,
                max_retries=max_retries,
                scaling_factor=scaling_factor,
            )

            print("Wave ", i, self.simulator.param_bounds)

            if len(test_x) < n_simulations or len(impl_scores) < n_simulations:
                msg = (
                    f"Not enough parameters or impl scores generated in wave {i + 1}",
                    f"/{n_waves}. Stopping history matching workflow. Results are ",
                    f"stored until wave {i}/{n_waves}.",
                )
                logger.warning(msg)
                break

            wave_results.append((test_x, impl_scores))

            # Get NROY points from impl scores and check fraction
            nroy_x = self.get_nroy(impl_scores, test_x)
            nroy_frac = nroy_x.shape[0] / test_x.shape[0]
            logger.info(
                " Wave %d/%d: NROY fraction is %.2f%%",
                i + 1,
                n_waves,
                nroy_frac * 100,
            )
            if nroy_frac > frac_nroy_stop:
                logger.info(
                    " Stopping history matching workflow at wave %d/%d "
                    "with NROY fraction %.2f%% < %.2f%%",
                    i + 1,
                    n_waves,
                    nroy_frac * 100,
                    frac_nroy_stop * 100,
                )
                break

        return wave_results
