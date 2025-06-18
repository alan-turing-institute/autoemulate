import torch

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators import GaussianProcessExact
from autoemulate.experimental.simulations.base import Simulator
from autoemulate.experimental.types import DeviceLike, GaussianLike, TensorLike


class HistoryMatching(TorchDeviceMixin):
    """
    History matching is a model calibration method, which uses observed data to
    rule out ``implausible`` parameter values. The implausibility metric is:

    .. math::
        I_i(\bar{x_0}) = \frac{|z_i - \\mathbb{E}(f_i(\bar{x_0}))|}
        {\\sqrt{\text{Var}[z_i - \\mathbb{E}(f_i(\bar{x_0}))]}}

    Query points above a given implausibility threshold are ruled out (RO)
    whereas all other points are marked as not ruled out yet (NROY).
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
            default val of ``1`` indicates that the largest implausibility will be used.
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
            Tensor indicating whether each parameter point is NROY given
            self.rank and self.threshold values.
        """
        # Sort implausibilities for each sample (descending)
        I_sorted, _ = torch.sort(implausibility, dim=1, descending=True)
        # The rank-th highest implausibility must be <= threshold
        return I_sorted[:, self.rank - 1] <= self.threshold

    def get_nroy(
        self, implausibility: TensorLike, x: TensorLike | None = None
    ) -> TensorLike:
        """
        Get indices of NROY points from implausibility scores. If `x`
        is provided, returns parameter values at NROY indices.

        Parameters
        ----------
        implausibility: TensorLike
            Tensor of implausibility scores for tested input parameters.
        x: Tensorlike | None
            Optional tensor of input parameters.

        Returns
        -------
        TensorLike
            Indices of NROY points or `x` at NROY indices.
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
        Get indices of RO points from implausibility scores. If `x`
        is provided, returns parameter values at RO indices.

        Parameters
        ----------
        implausibility: TensorLike
            Tensor of implausibility scores for tested input parameters.
        x: Tensorlike | None
            Optional tensor of iput parameters.

        Returns
        -------
        TensorLike
            Indices of RO points or `x` at RO indices.
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


class HistoryMatchingWorkflow(HistoryMatching):
    """
    Run history matching workflow:
    - sample parameter values to test from the current NROY parameter space
    - use emulator to filter out implausible samples and update NROY space
    - make predictions for a subset the NROY parameters using the simulator
    - refit the emulator using the simulated data
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        simulator: Simulator,
        # TODO: make this EmulatorWithUncertainty once implemented (see #542)
        emulator: GaussianProcessExact,
        observations: dict[str, tuple[float, float]] | dict[str, float],
        threshold: float = 3.0,
        model_discrepancy: float = 0.0,
        rank: int = 1,
        device: DeviceLike | None = None,
        train_x: TensorLike | None = None,
        train_y: TensorLike | None = None,
    ):
        """
        Initialize the history matching workflow object.

        TODO:
        - add random seed for reproducibility (once #465 is complete)

        Parameters
        ----------
        simulator: Simulator
            A simulator.
        emulator: GaussianProcessExact
            TODO: make this EmulatorWithUncertainty once implemented (see #542)
            A Gaussian Process emulator pre-trained on `simulator` data.
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
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.
        train_x: TensorLike | None
            Optional tensor of input data the emulator was trained on.
        train_y: TensorLike | None
            Optional tensor of output data the emulator was trained on.
        """
        super().__init__(observations, threshold, model_discrepancy, rank, device)
        self.simulator = simulator
        self.emulator = emulator
        self.emulator.device = self.device

        # These get updated when run() is called and used to refit the emulator
        if train_x is not None and train_y is not None:
            self.train_x = train_x
            self.train_y = train_y
        else:
            self.train_x = torch.empty((0, self.simulator.in_dim), device=self.device)
            self.train_y = torch.empty((0, self.simulator.out_dim), device=self.device)

    def _emulator_predict(self, x: TensorLike) -> tuple[TensorLike, TensorLike]:
        """
        Return emulator mean and variance prediction for input parameters `x`.
        """
        output = self.emulator.predict(x)
        assert isinstance(output, GaussianLike)
        assert output.variance.ndim == 2
        return (
            output.mean.float().detach(),
            output.variance.float().detach(),
        )

    def run(
        self, n_simulations: int = 100, n_test_samples=10000
    ) -> tuple[TensorLike, TensorLike]:
        """
        Run the iterative history matching workflow.

        Parameters
        ----------
        n_simulations: int
            Number of simulations to run.
        n_test_samples: int
            Number of input parameters to test for implausibility with the emulator.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tensor of tested input parameters and their imaplausability scores from
            which simulation samples were then selected.
        """

        # Generate `n_test_samples` of NROY parameters
        test_x = self.simulator.sample_inputs(n_test_samples)

        # Rule out implausible parameters from samples using an emulator
        pred_means, pred_vars = self._emulator_predict(test_x)
        impl_scores = self.calculate_implausibility(pred_means, pred_vars)
        nroy_x = self.get_nroy(impl_scores, test_x)

        # Update parameter bounds to NROY -> next time sample from this region
        if nroy_x.shape[0] > 1:
            min_nroy_values = torch.min(nroy_x, dim=0).values
            max_nroy_values = torch.max(nroy_x, dim=0).values
            self.simulator._param_bounds = list(
                zip(min_nroy_values.tolist(), max_nroy_values.tolist(), strict=False)
            )

        # Randomly pick `n_simulation_samples` from NROY to simulate predictions for
        # TODO: what if this returns fewer than `n_simulation_samples`?
        # - keep sampling if have fewer NROY samples (should this be random/uniform?)
        # - append test_x and impl_scores
        idx = torch.randperm(nroy_x.shape[0])[:n_simulations]
        to_simulate_x = nroy_x[idx]
        results = self.simulator.forward_batch(to_simulate_x)

        # Filter out parameters that simulator failed to return predictions for
        # TODO: this assumes that simulator returns None if it fails (see #438)
        valid_indices = [i for i, res in enumerate(results) if res is not None]
        simulated_x = to_simulate_x[valid_indices]
        pred_means = results[valid_indices]

        # Refit emulator
        self.train_y = torch.cat([self.train_y, pred_means], dim=0)
        self.train_x = torch.cat([self.train_x, simulated_x], dim=0)
        self.emulator.fit(self.train_x, self.train_y)

        # Return test parameters and impl scores for this run/wave
        return test_x, impl_scores
