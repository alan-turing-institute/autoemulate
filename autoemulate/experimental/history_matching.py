from typing import Optional, Union

import torch
from tqdm import tqdm

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators import GaussianProcessExact
from autoemulate.experimental.simulations.base import Simulator, SimulatorMetadata
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
        simulator: Simulator | SimulatorMetadata,
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
        - update to make sure we serve all the expected workflows
            - only filter candidate samples if simulate outputs but have
              an emulator available for this
            - alternatively just simulate OR emulate everything
            - also want to be able to use HM without either (already have
              some predictions available)

        Parameters
        ----------
        simulator: Simulator | SimulatorMetadata
            A simulator or the simulation metadata.
        observations: dict[str, tuple[float, float] | dict[str, float]
            For each output variable, specifies observed [value, noise]. In case
            of no uncertainty in observations, provides just the observation.
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

        self.simulator = simulator
        self.threshold = threshold
        self.discrepancy = model_discrepancy

        if rank > self.simulator.out_dim:
            raise ValueError(
                f"Rank {rank} is more than the simulator output dimension of ",
                f"{self.simulator.out_dim}",
            )
        self.rank = rank

        # Save mean and variance of observations
        if not set(observations.keys()).issubset(set(self.simulator.output_names)):
            raise ValueError(
                f"Observation keys {set(observations.keys())} must be a subset of ",
                f"simulator output names {set(self.simulator.output_names)}",
            )

        # Shape: [1, n_outputs]
        self.obs_means, self.obs_vars = self._process_observations(observations)

        # Track tested parameter values and their implausability scores
        self.tested_params = torch.empty((0, self.simulator.in_dim), device=self.device)
        self.impl_scores = torch.empty((0, self.simulator.out_dim), device=self.device)

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
            of no uncertainty in observations, provides just the observation.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            Tensors of observations and the associated noise (which can be 0).
        """
        means = []
        variances = []

        for key, value in observations.items():
            if isinstance(value, float):  # Single float case
                means.append(value)
                variances.append(0.0)
            elif isinstance(value, tuple) and len(value) == 2:  # Tuple case
                mean, variance = value
                means.append(mean)
                variances.append(variance)
            else:
                raise ValueError(f"Invalid observation format for key '{key}': {value}")

        # Convert lists to tensors
        means_tensor = torch.tensor(means, dtype=torch.float32)
        variances_tensor = torch.tensor(variances, dtype=torch.float32)

        # Reshape observation tensors for broadcasting
        return means_tensor.view(1, -1), variances_tensor.view(1, -1)

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
            torch.rand((n_samples, self.simulator.in_dim), device=self.device)
            * (max_bounds - min_bounds)
            + min_bounds
        )

    def sample_params(self, n_samples: int, nroy_samples: Optional[TensorLike] = None):
        """
        Generate new parameter samples, either using `self.simulator` or from
        within NROY space if `nroy_samples` is provided and it is not empty.

        Parameters
        ----------
            n_samples: int
                Number of new samples to generate.
            nroy_samples: TensorLike | None
                Optional tensor of parameter samples in the NROY space. If
                provided, sample from within this space.

        Returns
        -------
            TensorLike
                Tensor of parameter samples [n_samples, n_parameters].
        """
        # TODO: remove numpy conversions once merged with #414
        if nroy_samples is None or nroy_samples.size(0) == 0:
            return self.simulator.sample_inputs(n_samples)
        return self.sample_nroy(n_samples, nroy_samples)

    def predict(
        self,
        x: TensorLike,
        emulator: Optional[GaussianProcessExact] = None,
    ) -> tuple[TensorLike, Optional[TensorLike], TensorLike]:
        """
        Make predictions for a batch of inputs x. Uses `self.simulator` unless
        an emulator trained on `self.simulator` data is provided.

        Parameters
        ----------
        x: TensorLike
            Tensor of parameters to simulate/emulate returned by `self.sample`.
        emulator: GaussianProcessExact | None
            NOTE: this can be other GP emulators when implemented.
            Gaussian process emulator trained on `self.simulator` output data.

        Returns
        -------
        tuple[TensorLike, TensorLike | None, TensorLike]
            Arrays of predicted means and optionally variances as well as the input
            data for which predictions were made succesfully.
        """
        if x.shape[0] == 0:
            return (
                torch.empty((0, self.simulator.out_dim), device=self.device),
                torch.empty((0, self.simulator.out_dim), device=self.device),
                torch.empty((0, self.simulator.in_dim), device=self.device),
            )
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
        elif isinstance(self.simulator, Simulator):
            # Simulator is determinstic, have no predictive variance
            pred_vars = None

            # Simulator returns None if it fails, discard these runs and inputs
            results = self.simulator.forward_batch(x)
            valid_indices = [i for i, res in enumerate(results) if res is not None]
            pred_means, x = results[valid_indices], x[valid_indices]

        else:
            msg = "Need an Emulator or a Simulator to make predictions."
            raise ValueError(msg)

        return pred_means, pred_vars, x

    def run(
        self,
        n_waves: int = 1,
        n_samples_per_wave: int = 100,
        emulator_predict: bool = True,
        emulator: Optional[GaussianProcessExact] = None,
    ) -> Union[GaussianProcessExact, None]:
        """
        Run iterative history matching. In each wave:
            - sample parameter values to test from the NROY space
                - at the start, NROY is the entire parameter space
            - make predictions for the sampled parameters:
                - either, use the provided emulator to make predictions
                - or, use `self.simulator` to make predictions and update
                  the emulator after each wave (if there are enough
                  succesful samples)
            - compute implausability scores for predictions and further
              reduce NROY space

        Parameters
        ----------
        n_waves: int
            Number of iterations of history matching to run.
        n_samples_per_wave: int
            Number of parameter samples to make predictions for in each wave.
        emulator_predict: bool = True
            Whether to use the emulator to make predictions. If False, use
            `self.simulator` instead.
        emulator: GaussianProcessExact | None
            NOTE: this can be other GP emulators when implemented.
            Gaussian Process emulator pre-trained on `self.simulator` data.
            - if `emulator_predict=True`, the GP is used to make predictions.
            - if `emulator_predict=False`, `self.simulator` is used to make
              predictions and the GP is retrained on the simulated data.

        Returns
        -------
        union[GaussianProcessExact, None]
            - a GP emulator (retrained on new data if `emulator_predict=False`) or None
        """
        if emulator_predict and emulator is None:
            msg = "Need to pass a GP emulator object when `emulator_predict=True`"
            raise ValueError(msg)

        if emulator is not None:
            emulator.device = self.device

        # Keep track of predictions in case refitting emulator
        ys = torch.empty(0, device=self.device)

        # To begin with entire parameter space is NROY so don't have samples yet
        nroy_samples = None

        with tqdm(
            total=n_waves,
            desc="History Matching",
            unit="wave",
            disable=self.device.type != "cpu",
        ) as pbar:
            for _ in range(n_waves):
                samples = self.sample_params(n_samples_per_wave, nroy_samples)

                # Filter out RO samples (if have simulator and an emulator)
                # this is quite similar to an active learning workflow
                if (not emulator_predict) and (emulator is not None):
                    pred_means, pred_vars, _ = self.predict(samples)
                    impl_scores = self.calculate_implausibility(pred_means, pred_vars)
                    nroy_idx = self.get_nroy(impl_scores)
                    samples[nroy_idx]

                # Emulate (or simulate) predictions
                pred_means, pred_vars, successful_samples = self.predict(
                    x=samples,
                    emulator=emulator if emulator_predict else None,
                )

                # Calculate implausibility and identify NROY points
                impl_scores = self.calculate_implausibility(pred_means, pred_vars)
                nroy_idx = self.get_nroy(impl_scores)
                nroy_samples = successful_samples[nroy_idx]

                # Store results
                self.tested_params = torch.cat(
                    [self.tested_params, successful_samples], dim=0
                )
                self.impl_scores = torch.cat([self.impl_scores, impl_scores], dim=0)

                # Refit emulator
                if (not emulator_predict) and (emulator is not None):
                    ys = torch.cat([ys, pred_means], dim=0)
                    emulator.fit(self.tested_params, ys)

                # Update progress bar
                self._update_progress_bar(
                    pbar, impl_scores, samples.size(0), nroy_samples.size(0)
                )
                pbar.update(1)

        return emulator

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
