from typing import Optional, Union

import torch
from tqdm import tqdm

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

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
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
            TODO: should this just be a 1D or 2D tensor of values
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
            raise ValueError(
                "Observations must be either float or tuple of two floats."
            )

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
        if self.rank == 1:
            # First-order implausibility: all outputs must satisfy threshold
            return torch.all(self.threshold >= implausability, dim=1)
        # Sort implausibilities for each sample (descending)
        I_sorted, _ = torch.sort(implausability, dim=1, descending=True)
        # The rank-th highest implausibility must be <= threshold
        return I_sorted[:, self.rank - 1] <= self.threshold

    def get_nroy(self, implausability: TensorLike) -> TensorLike:
        """
        Get indices of NROY points from implausability scores.

        Parameters
        ----------
        implausability: TensorLike
            Tensor of implausability scores for tested parameters.

        Returns
        -------
        TensorLike
            Indices of NROY points.
        """
        nroy_mask = self._create_nroy_mask(implausability)
        return torch.where(nroy_mask)[0]

    def get_ro(self, implausability: TensorLike) -> TensorLike:
        """
        Get indices of RO points from implausability scores.

        Parameters
        ----------
        implausability: TensorLike
            Tensor of implausability scores for tested parameters.

        Returns
        -------
        TensorLike
            Indices of RO points.
        """
        nroy_mask = self._create_nroy_mask(implausability)
        return torch.where(~nroy_mask)[0]

    def calculate_implausibility(
        self,
        pred_means: TensorLike,  # [n_samples, n_outputs]
        pred_vars: Optional[TensorLike] = None,  # [n_samples, n_outputs]
        default_var: float = 0.01,
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
        default_var: int
            TODO: should this just be 0?
            Prediction variance value to use if not provided.

        Returns
        -------
        TensorLike
            Tensor of implausibility scores.
        """
        # Set prediction variances if not provided
        if pred_vars is None:
            pred_vars = torch.full_like(pred_means, default_var, device=self.device)

        # Additional variance due to model discrepancy (defaults to 0)
        discrepancy = torch.full_like(
            self.obs_vars, self.discrepancy, device=self.device
        )

        # Calculate total variance
        Vs = pred_vars + discrepancy + self.obs_vars

        # Calculate implausibility
        return torch.abs(self.obs_means - pred_means) / torch.sqrt(Vs)

    def filter_nroy_samples(
        self, samples: TensorLike, pred_means: TensorLike, pred_vars: TensorLike
    ) -> TensorLike:
        """
        Given input parameter samples and predicted means and variances for
        each input, return only NROY input parameter samples.

        Parameters
        ----------
        samples: TensorLike
            Tensor of input parameters used to make predictions.
        pred_means: TensorLike
            Tensor of prediction means [n_samples, n_outputs]
        pred_vars: TensorLike | None
            Tensor of prediction variances [n_samples, n_outputs]. If not
            provided (e.g., when predictions are made by a deterministic
            simulator), all variances are set to `default_var`.

        Returns
        -------
        TensorLike
            Tensor of parameter samples [n_samples, n_parameters] within
            NROY.
        """
        impl_scores = self.calculate_implausibility(pred_means, pred_vars)
        nroy_idx = self.get_nroy(impl_scores)
        return samples[nroy_idx]

    def sample_nroy(
        self,
        n_samples: int,
        nroy_samples: TensorLike,
    ) -> TensorLike:
        """
        Generate new parameter samples within NROY space.

        Parameters
        ----------
        n_samples: int
            Number of new samples to generate within the NROY bounds.
        nroy_samples: TensorLike
            Tensor of parameter samples in the NROY space [samples, parameters].

        Returns
        -------
        TensorLike
            Tensor of parameter samples [n_samples, n_parameters].
        """

        min_bounds, _ = torch.min(nroy_samples, dim=0)
        max_bounds, _ = torch.max(nroy_samples, dim=0)
        return (
            torch.rand((n_samples, nroy_samples.shape[1]), device=self.device)
            * (max_bounds - min_bounds)
            + min_bounds
        )

    def _predict(
        self,
        x: TensorLike,
        simulator: Optional[Simulator] = None,
        emulator: Optional[GaussianProcessExact] = None,
    ) -> tuple[TensorLike, Optional[TensorLike], TensorLike]:
        """
        Make predictions for a batch of inputs x. Uses `simulator` unless
        an emulator trained on `simulator` data is provided.

        Parameters
        ----------
        x: TensorLike
            Tensor of parameters to simulate/emulate returned by `self.sample`.
        simulator: Simulator | None
            An optional simulator. Must be provided if `emulator=None`.
        emulator: GaussianProcessExact | None
            NOTE: this can be other GP emulators when implemented.
            An optional Gaussian Process emulator pre-trained on `simulator` data.
            Must be provided if `simulator=None`.

        Returns
        -------
        tuple[TensorLike, TensorLike | None, TensorLike]
            Arrays of predicted means and optionally variances as well as the input
            data for which predictions were made succesfully.
        """
        if x.shape[0] == 0:
            return (
                torch.empty((0, simulator.out_dim), device=self.device),
                torch.empty((0, simulator.out_dim), device=self.device),
                torch.empty((0, simulator.in_dim), device=self.device),
            )

        # TODO: if both emulator and simulator are provided, uses simulator
        # - is this expected behaviour?
        # - should we handle this?
        # Make predictions using a GP emulator
        if emulator is not None:
            output = emulator.predict(x)
            assert isinstance(output, GaussianLike)
            assert output.variance.ndim == 2
            pred_means, pred_vars = (
                output.mean.float().detach(),
                output.variance.float().detach(),
            )

        # Make predictions using the simulator
        elif simulator is not None:
            # Simulator is determinstic, have no predictive variance
            pred_vars = None

            # Simulator returns None if it fails, discard these runs and inputs
            results = simulator.forward_batch(x)
            valid_indices = [i for i, res in enumerate(results) if res is not None]
            pred_means, x = results[valid_indices], x[valid_indices]
        else:
            msg = "Either an emulator or a simulator must be provided."
            raise ValueError(msg)

        return pred_means, pred_vars, x

    def run(
        self,
        simulator: Simulator,
        emulator: GaussianProcessExact,
        n_waves: int = 1,
        n_samples_per_wave: int = 100,
    ) -> tuple[TensorLike, TensorLike, GaussianProcessExact | None]:
        """
        Run iterative history matching. In each wave:
            - sample parameter values to test from the NROY space
                - at the start, NROY is the entire parameter space
                - use emulator to filter out implausible samples
            - make predictions for the sampled parameters using the simulator
            - refit the emulator using the simulated data

        Parameters
        ----------
        simulator: Simulator
            A simulator.
        emulator: GaussianProcessExact
            NOTE: this can be other GP emulators when implemented.
            A Gaussian Process emulator pre-trained on `simulator` data.
        n_waves: int
            Number of iterations of history matching to run.
        n_samples_per_wave: int
            Number of parameter samples to make predictions for in each wave.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            Simulated parameters and their implausability scores.
        """
        if emulator is not None:
            emulator.device = self.device

        # TODO: we should keep track of these within the class
        # Keep track of predictions in case refitting emulator
        tested_params = torch.empty((0, simulator.in_dim), device=self.device)
        ys = torch.empty((0, simulator.out_dim), device=self.device)
        impl_scores = torch.empty((0, simulator.out_dim), device=self.device)

        # To begin with entire parameter space is NROY so don't have samples yet
        nroy_samples = None

        # TODO: add logging
        with tqdm(
            total=n_waves,
            desc="History Matching",
            unit="wave",
            disable=self.device.type != "cpu",
        ) as pbar:
            for _ in range(n_waves):
                if nroy_samples is None or nroy_samples.size(0) == 0:
                    samples = simulator.sample_inputs(n_samples_per_wave)
                samples = self.sample_nroy(n_samples_per_wave, nroy_samples)

                # Filter out RO samples (if have an emulator)
                if emulator is not None:
                    pred_means, pred_vars, _ = self._predict(samples, emulator)
                    samples = self.filter_nroy_samples(samples, pred_means, pred_vars)

                # Simulate predictions
                pred_means, pred_vars, successful_samples = self._predict(
                    x=samples,
                    emulator=None,
                )

                # Calculate implausibility and identify NROY points
                impl_scores = self.calculate_implausibility(pred_means, pred_vars)
                nroy_idx = self.get_nroy(impl_scores)
                nroy_samples = successful_samples[nroy_idx]

                # Store results
                tested_params = torch.cat([tested_params, successful_samples], dim=0)
                impl_scores = torch.cat([impl_scores, impl_scores], dim=0)

                # Refit emulator
                if emulator is not None:
                    ys = torch.cat([ys, pred_means], dim=0)
                    emulator.fit(tested_params, ys)

                # Update progress bar
                self._update_progress_bar(
                    pbar, impl_scores, samples.size(0), nroy_samples.size(0)
                )
                pbar.update(1)

        # TODO: don't love returning all of these - can we store them somewhere?
        # - maybe create placeholders for them in init
        return tested_params, impl_scores, emulator

    def _update_progress_bar(
        self, pbar: tqdm, impl_scores: TensorLike, n_samples: int, n_nroy_samples: int
    ):
        """
        Updates the progress bar.

        Parameters
        ----------
        pbar: tqdm
            The progress bar.
        impl_scores: TensorLike
            Tensor of implausibility scores for succesful parameter samples.
        n_samples: int
            Total number of tested parameter samples.
        n_nroy_samples: int
            Number of parameter samples in the NROY space.
        """
        failed_count = n_samples - impl_scores.size(0)
        min_impl = torch.min(impl_scores) if impl_scores.size(0) > 0 else "NaN"
        max_impl = torch.max(impl_scores) if impl_scores.size(0) > 0 else "NaN"
        pbar.set_postfix(
            {
                "samples": n_samples,
                "failed": failed_count,
                "NROY": n_nroy_samples,
                "min_impl": f"{min_impl:.2f}",
                "max_impl": f"{max_impl:.2f}",
            }
        )
