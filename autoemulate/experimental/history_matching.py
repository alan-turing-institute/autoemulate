from typing import Optional, Union

import torch

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators import GaussianProcessExact
from autoemulate.experimental.simulations.base import Simulator
from autoemulate.experimental.types import DeviceLike, GaussianLike, TensorLike


class HistoryMatching(TorchDeviceMixin):
    """
    History matching is a model calibration method, which uses observed data to
    rule out ``implausible`` parameter values. The implausability metric is:

    .. math::
        I_i(\bar{x_0}) = \frac{|z_i - \\mathbb{E}(f_i(\bar{x_0}))|}
        {\\sqrt{\text{Var}[z_i - \\mathbb{E}(f_i(\bar{x_0}))]}}

    Query points above a given implausibility threshold are ruled out (RO)
    whereas all other points are marked as not ruled out yet (NROY).
    """

    def __init__(
        self,
        observations: Union[dict[str, tuple[float, float]], dict[str, float]],
        threshold: float = 3.0,
        model_discrepancy: float = 0.0,
        rank: int = 1,
        device: DeviceLike | None = None,
    ):
        """
        Initialize the history matching object.

        TODO:
        - add random seed (once #465 is complete)

        Parameters
        ----------
        observations: dict[str, tuple[float, float] | dict[str, float]
            For each output variable, specifies observed [value, noise]. In case
            of no uncertainty in observations, provides just the observed value.
        threshold: float
            Implausibility threshold (query points with implausability scores that
            exceed this value are ruled out). Defaults to 3, which is considered
            a good value for simulations with a single output.
        model_discrepancy: float
            Additional variance to include in the implausability calculation.
        rank: int
            Scoring method for multi-output problems. Must be a non-negative
            integer less than the number of outputs. When the implausability
            scores are ordered across outputs, it indicates which rank to use
            when determining whether the query point is NROY. The default value
            of ``1`` indicates that the largest implausibility will be used.
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.
        """
        TorchDeviceMixin.__init__(self, device=device)

        self.threshold = threshold
        self.discrepancy = model_discrepancy
        self.out_dim = len(observations)

        if rank > self.out_dim:
            raise ValueError(
                f"Rank {rank} is more than the simulator output dimension of ",
                f"{self.out_dim}",
            )
        self.rank = rank

        # Save mean and variance of observations, shape: [1, n_outputs]
        self.obs_means, self.obs_vars = self._process_observations(observations)

    def _process_observations(
        self,
        observations: Union[dict[str, tuple[float, float]], dict[str, float]],
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
        values = torch.tensor(list(observations.values()))

        # No variance
        if values.ndim == 1:
            means = values
            variances = torch.zeros_like(means)
        # Values are (mean, variance)
        elif values.ndim == 2:
            means = values[:, 0]
            variances = values[:, 1]
        else:
            msg = "Observations must be either float or tuple of two floats."
            raise ValueError(msg)

        # Reshape observation tensors for broadcasting
        return means.view(1, -1), variances.view(1, -1)

    def _create_nroy_mask(self, implausability: TensorLike) -> TensorLike:
        """
        Create mask for NROY points based on rank.

        Parameters
        ----------
        implausability: TensorLike
            Tensor of implausability scores for tested parameters.

        Returns
        -------
        TensorLike
            Tensor indicating whether each parameter point is NROY given
            self.rank and self.threshold values.
        """
        # Sort implausibilities for each sample (descending)
        I_sorted, _ = torch.sort(implausability, dim=1, descending=True)
        # The rank-th highest implausibility must be <= threshold
        return I_sorted[:, self.rank - 1] <= self.threshold

    def get_nroy(
        self, implausability: TensorLike, parameters: Optional[TensorLike] = None
    ) -> TensorLike:
        """
        Get indices of NROY points from implausability scores. If `parameters`
        is provided, returns parameter values at NROY indices.

        Parameters
        ----------
        implausability: TensorLike
            Tensor of implausability scores for tested parameters.
        parameters: Tensorlike | None
            Optional tensor of parameters.

        Returns
        -------
        TensorLike
            Indices of NROY points or `parameters` at NROY indices.
        """
        nroy_mask = self._create_nroy_mask(implausability)
        idx = torch.where(nroy_mask)[0]
        if parameters is None:
            return idx
        return parameters[idx]

    def get_ro(
        self, implausability: TensorLike, parameters: Optional[TensorLike] = None
    ) -> TensorLike:
        """
        Get indices of RO points from implausability scores. If `parameters`
        is provided, returns parameter values at RO indices.

        Parameters
        ----------
        implausability: TensorLike
            Tensor of implausability scores for tested parameters.
        parameters: Tensorlike | None
            Optional tensor of parameters.

        Returns
        -------
        TensorLike
            Indices of RO points or `parameters` at RO indices.
        """
        nroy_mask = self._create_nroy_mask(implausability)
        idx = torch.where(~nroy_mask)[0]
        if parameters is None:
            return idx
        return parameters[idx]

    def calculate_implausibility(
        self,
        pred_means: TensorLike,  # [n_samples, n_outputs]
        pred_vars: Optional[TensorLike] = None,  # [n_samples, n_outputs]
    ) -> TensorLike:
        """
        Calculate implausibility scores.

        Parameters
        ----------
        pred_means: TensorLike
            Tensor of prediction means [n_samples, n_outputs]
        pred_vars: TensorLike | None
            Tensor of prediction variances [n_samples, n_outputs]. If not
            provided (e.g., when predictions are made by a deterministic
            simulator), all variances are set to `default_var`.

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


class HistoryMatchingWorkflow(HistoryMatching):
    """
    Run history matching workflow:
    - sample parameter values to test from the NROY space
        - at the start, NROY is the entire parameter space
    - use emulator to filter out implausible samples
    - make predictions for the sampled parameters using the simulator
    - refit the emulator using the simulated data
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        simulator: Simulator,
        emulator: GaussianProcessExact,
        observations: Union[dict[str, tuple[float, float]], dict[str, float]],
        threshold: float = 3.0,
        model_discrepancy: float = 0.0,
        rank: int = 1,
        device: DeviceLike | None = None,
    ):
        """
        Initialize the history matching workflow object.

        Parameters
        ----------
        simulator: Simulator
            A simulator.
        emulator: GaussianProcessExact
            NOTE: this can be other GP emulators when implemented.
            A Gaussian Process emulator pre-trained on `simulator` data.
        observations: dict[str, tuple[float, float] | dict[str, float]
            For each output variable, specifies observed [value, noise]. In case
            of no uncertainty in observations, provides just the observed value.
        threshold: float
            Implausibility threshold (query points with implausability scores that
            exceed this value are ruled out). Defaults to 3, which is considered
            a good value for simulations with a single output.
        model_discrepancy: float
            Additional variance to include in the implausability calculation.
        rank: int
            Scoring method for multi-output problems. Must be a non-negative
            integer less than the number of outputs. When the implausability
            scores are ordered across outputs, it indicates which rank to use
            when determining whether the query point is NROY. The default value
            of ``1`` indicates that the largest implausibility will be used.
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.
        """
        super().__init__(observations, threshold, model_discrepancy, rank, device)
        self.simulator = simulator
        self.emulator = emulator
        self.emulator.device = self.device

        # These get populated when run() is called
        self.tested_params = torch.empty((0, len(observations)), device=self.device)
        self.impl_scores = torch.empty((0, len(observations)), device=self.device)
        self.ys = torch.empty((0, self.simulator.out_dim), device=self.device)

    def run(self, n_samples: int = 100):
        """
        Run the iterative history matching workflow.

        Parameters
        ----------
        n_samples: int
            Number of parameter samples to make predictions for.
        """

        # Sample from the NROY parameter space - to begin with this is the entire space
        parameter_samples = self.simulator.sample_inputs(n_samples)

        # Rule out implausible parameters from samples using an emulator
        output = self.emulator.predict(parameter_samples)
        assert isinstance(output, GaussianLike)
        assert output.variance.ndim == 2
        pred_means, pred_vars = (
            output.mean.float().detach(),
            output.variance.float().detach(),
        )
        impl_scores = self.calculate_implausibility(pred_means, pred_vars)
        parameter_samples = self.get_nroy(impl_scores, parameter_samples)

        # Simulate predictions (predicted variance is None)
        results = self.simulator.forward_batch(parameter_samples)
        pred_vars = None

        # This assumes that simulator returns None if it fails
        # TODO: update as part of #438
        valid_indices = [i for i, res in enumerate(results) if res is not None]
        successful_parameter_samples = parameter_samples[valid_indices]
        pred_means = results[valid_indices]

        # Calculate implausibility and store results
        impl_scores = self.calculate_implausibility(pred_means, pred_vars)
        self.tested_params = torch.cat(
            [self.tested_params, successful_parameter_samples], dim=0
        )
        self.impl_scores = torch.cat([self.impl_scores, impl_scores], dim=0)

        # Restrict parameter bounds to sample from to NROY min/max
        nroy_parameter_samples = self.get_nroy(
            impl_scores, successful_parameter_samples
        )
        if nroy_parameter_samples.shape[0] > 1:
            min_nroy_values = torch.min(nroy_parameter_samples, dim=0).values
            max_nroy_values = torch.max(nroy_parameter_samples, dim=0).values
            self.simulator._param_bounds = list(
                zip(min_nroy_values.tolist(), max_nroy_values.tolist())
            )

        # Refit emulator
        self.ys = torch.cat([self.ys, pred_means], dim=0)
        self.emulator.fit(self.tested_params, self.ys)
