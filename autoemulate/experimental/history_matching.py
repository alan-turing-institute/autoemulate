from typing import Optional, Union

import torch
from tqdm import tqdm

from autoemulate.experimental.types import TensorLike
from autoemulate.simulations.base import Simulator


# TODO: should we use ValidationMixin here?
class HistoryMatching:
    """
    History matching is a model calibration method, which uses observed data to rule out
    parameter values which are ``implausible``. The implausability metric is:

    .. math::
        I_i(\bar{x_0}) = \frac{|z_i - \\mathbb{E}(f_i(\bar{x_0}))|}
        {\\sqrt{\text{Var}[z_i - \\mathbb{E}(f_i(\bar{x_0}))]}}

    Query points above a given implausibility threshold are ruled out (RO) whereas
    all other points are marked as not ruled out yet (NROY).
    """

    def __init__(
        self,
        simulator: Simulator,
        observations: dict[str, tuple[float, float]],
        threshold: float = 3.0,
        model_discrepancy: float = 0.0,
        rank: int = 1,
    ):
        """
        Initialize the history matching object.

        TODO:
        - make this work with experimental GP emulators
        - make this work with updated Simulator class (after #414 is merged)
        - add device handling
        - add random seed following #479

        Parameters
        ----------
            simulator: Simulator
                The simulation to emulate.
            observations: dict[str, tuple[float, float]]
                For each output variable, specifies observed [mean, variance] values.
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
        """
        self.simulator = simulator
        self.threshold = threshold
        self.discrepancy = model_discrepancy

        # TODO: should this be in Simulator?
        self.in_dim = len(self.simulator.param_names)
        self.out_dim = len(self.simulator.output_names)

        if rank > self.out_dim:
            raise ValueError(
                f"Rank {rank} is more than the simulator output dimension of ",
                f"{self.out_dim}",
            )
        self.rank = rank

        # Save mean and variance of observations
        if not set(observations.keys()).issubset(set(self.simulator.output_names)):
            raise ValueError(
                f"Observation keys {set(observations.keys())} must be a subset of ",
                f"simulator output names {set(self.simulator.output_names)}",
            )
        obs_means = torch.tensor(
            [observations[name][0] for name in self.simulator.output_names]
        )
        obs_vars = torch.tensor(
            [observations[name][1] for name in self.simulator.output_names]
        )
        # Reshape observation arrays for broadcasting
        self.obs_means = obs_means.view(1, -1)  # [1, n_outputs]
        self.obs_vars = obs_vars.view(1, -1)  # [1, n_outputs]

        # Quantities to track
        self.tested_params = torch.empty((0, self.out_dim))
        self.impl_scores = torch.empty((0, self.out_dim))

    def calculate_implausibility(
        self,
        pred_means: TensorLike,  # [n_samples, n_outputs]
        pred_vars: Optional[TensorLike] = None,  # [n_samples, n_outputs]
        default_var: float = 0.01,
    ) -> dict[str, TensorLike]:
        """
        Calculate implausibility scores and identify RO and NROY points.

        Parameters
        ----------
            pred_means: TensorLike
                Tensor of prediction means [n_samples, n_outputs]
            pred_vars: optional TensorLike
                Tensor of prediction variances [n_samples, n_outputs]. If not
                provided (e.g., when predictions are made by a deterministic
                simulator), all variances are set to `default_var`.
            default_var: int
                Prediction variance to set if not provided.

        Returns
        -------
            dict[str, TensorLike]
                Contains the following key, value pairs:
                    - 'I': tensor of implausibility scores [n_samples, n_outputs]
                    - 'NROY': tensor of indices of Not Ruled Out Yet points
                    - 'RO': tensor of indices of Ruled Out points
        """
        # Set variances if not provided
        if pred_vars is None:
            pred_vars = torch.full_like(pred_means, default_var)

        # Add model discrepancy
        discrepancy = torch.full_like(self.obs_vars, self.discrepancy)

        # Calculate total variance
        Vs = pred_vars + discrepancy + self.obs_vars

        # Calculate implausibility
        I = torch.abs(self.obs_means - pred_means) / torch.sqrt(Vs)

        # Determine NROY points based on rank parameter
        if self.rank == 1:
            # First-order implausibility: all outputs must satisfy threshold
            nroy_mask = torch.all(self.threshold >= I, dim=1)
        else:
            # Sort implausibilities for each sample (descending)
            I_sorted = torch.sort(I, dim=1, descending=True)
            # The rank-th highest implausibility must be <= threshold
            nroy_mask = I_sorted[:, self.rank - 1] <= self.threshold

        return {
            "I": I,  # Implausibility scores
            "NROY": torch.where(nroy_mask)[0],  # Indices of NROY points
            "RO": torch.where(~nroy_mask)[0],  # Indices of RO points
        }

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

        # Need to handle possible discontinuous NROY spaces
        # i.e., a region within min/max bounds is not valid (RO)
        valid_samples = torch.empty((0, self.in_dim))
        while len(valid_samples) < n_samples:
            # Generate candidates
            candidate_samples = (
                torch.rand((n_samples, self.in_dim)) * (max_bounds - min_bounds)
                + min_bounds
            )

            # Filter valid samples based on implausibility and concatenate
            implausibility = self.calculate_implausibility(candidate_samples)
            valid_candidates = candidate_samples[implausibility["NROY"]]
            valid_samples = torch.cat([valid_samples, valid_candidates], dim=0)

            # Only return required number of samples
            if len(valid_samples) > n_samples:
                valid_samples = valid_samples[:n_samples]

        return valid_samples

    def sample_params(self, n_samples: int, nroy_samples: Optional[TensorLike] = None):
        """
        Generate new parameter samples, either using `self.simulator` or from
        within NROY space if `nroy_samples` is provided and it is not empty.

        Parameters
        ----------
            n_samples: int
                Number of new samples to generate.
            nroy_samples: optional[TensorLike]
                Optional tensor of parameter samples in the NROY space. If
                provided, sample from within this space.

        Returns
        -------
            TensorLike
                Tensor of parameter samples [n_samples, n_parameters].
        """
        # TODO: remove numpy conversions once merged with #414
        if nroy_samples is None or nroy_samples.size(0) == 0:
            samples = self.simulator.sample_inputs(n_samples)
            return torch.from_numpy(samples)
        return self.sample_nroy(n_samples, nroy_samples)

    def predict(
        self,
        x: TensorLike,
        # TODO: update emulator object passed here
        emulator: Optional[object] = None,
    ) -> tuple[TensorLike, TensorLike, TensorLike]:
        """
        Make predictions for a batch of inputs x. Uses `self.simulator` unless
        an emulator trained on `self.simulator` data is provided.

        Parameters
        ----------
        x: TensorLike
            Tensor of parameters to simulate/emulate returned by `self.sample`.
        TODO: update emulator type and description below
        emulator: optional object
            Gaussian process emulator trained on `self.simulator` output data.

        Returns
        -------
        tuple[TensorLike, TensorLike, TensorLike]
            Arrays of predicted means and variances as well as the input data for
            which predictions were made succesfully.
        """
        if x.shape[0] == 0:
            return torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

        # Make predictions using an emulator
        if emulator is not None:
            # TODO: remove numpy conversion
            pred_means, pred_stds = emulator.predict(x.numpy(), return_std=True)
            pred_vars = pred_stds**2
            # TODO: remove numpy conversion
            pred_means = torch.from_numpy(pred_means)
            pred_vars = torch.from_numpy(pred_vars)

            # TODO: don't need this once remove sklearn dependence
            # Ensure correct shape for single output case
            if len(pred_means.shape) == 1:
                pred_means = pred_means.view(-1, 1)
                pred_vars = pred_vars.view(-1, 1)

        # Make predictions using the simulator
        else:
            # TODO: remove numpy conversion
            results = self.simulator.run_batch_simulations(x.numpy())
            results = torch.from_numpy(results)

            # Filter out failed simulations
            # TODO: should the simulator handle this?
            valid_indices = [i for i, res in enumerate(results) if res is not None]
            x = x[valid_indices]
            pred_means = results[valid_indices]
            pred_vars = None
            if pred_means.shape[0] == 0:
                # All simulations failed
                return torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

        # Also return input vector in case simulation failed for some inputs
        return pred_means, pred_vars, x

    def run(
        self,
        n_waves: int = 3,
        n_samples_per_wave: int = 100,
        emulator_predict: bool = True,
        # TODO: update emulator type passed here
        initial_emulator: Optional[object] = None,
    ) -> tuple[TensorLike, TensorLike, Union[object, None]]:
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
        TODO: update emulator type and description below
        initial_emulator: optional object
            Gaussian Process emulator pre-trained on `self.simulator` data.
            - if `emulator_predict=True`, the GP is used to make predictions.
            - if `emulator_predict=False`, `self.simulator` is used to make
              predictions and the GP is retrained on the simulated data.

        Returns
        -------
        tuple[TensorLike, TensorLike, union[object, None]]
            - Tensor of all parameter samples for which predictions were made
            - Tensor of all implausability scores
            - a GP emulator (retrained on new data if `emulator_predict=False`) or None
        """
        if emulator_predict and initial_emulator is None:
            raise ValueError(
                "Need to pass a GP emulator object when `emulator_predict=True`"
            )

        # TODO: should emulator be passed at initialisation?
        emulator = initial_emulator
        current_samples = self.sample_params(n_samples_per_wave)

        # TODO: revisit where expect things to fail and handle appropriately
        # e.g., if succesful samples=0. will calculate_implausability handle it?
        # can we remove the if impl_scores.size(0) > 0 statement?
        with tqdm(total=n_waves, desc="History Matching", unit="wave") as pbar:
            for wave in range(n_waves):
                # Run wave using batch processing
                pred_means, pred_vars, successful_samples = self.predict(
                    x=current_samples,
                    # Emulate predictions unless emulator_predict=False
                    emulator=emulator if emulator_predict else None,
                )

                # Calculate implausibility in batch
                implausibility = self.calculate_implausibility(pred_means, pred_vars)

                # NROY parameters and implausibility scores for all parameters
                nroy_samples = successful_samples[implausibility["NROY"]]
                impl_scores = implausibility["I"]

                self.update_progress_bar(
                    pbar, impl_scores, current_samples.size(0), nroy_samples.size(0)
                )

                # Store results
                if impl_scores.size(0) > 0:
                    # Only include samples with scores
                    self.tested_params = torch.cat(
                        [self.tested_params, successful_samples], dim=0
                    )
                    self.impl_scores = torch.cat([self.impl_scores, impl_scores], dim=0)

                    # Update emulator if simulated (enough) data
                    if (not emulator_predict) and nroy_samples.size(0) > 10:
                        emulator = self.update_emulator(
                            emulator, successful_samples, pred_means
                        )

                # Generate new samples for next wave
                if wave < n_waves - 1:
                    current_samples = self.sample_params(
                        n_samples_per_wave, nroy_samples
                    )

                pbar.update(1)

        # TODO: maybe not return anything?
        return self.tested_params, self.impl_scores, emulator

    def update_emulator(
        self,
        # TODO: update emulator type passed here
        existing_emulator: object,
        x: TensorLike,
        y: TensorLike,
    ):
        """
        Update an existing GP emulator with new training data.

        Parameters
        ----------
            TODO: eventually this should be autoemulate.GaussianProcessExact
            existing_emulator: Gaussian Process from sklearn
                Trained GP emulator.
            x: TensorLike
                Tensor of parameter values to train emulator on.
            y: TensorLike
                Tensor of output values.

        Returns
        -------
            Updated GP emulator
        """
        # Instead of deepcopy, we'll create a new instance if needed
        # For now, just use the existing model as is
        updated_emulator = existing_emulator

        # Update the emulator
        try:
            # Refit the entire model, includes hyperparameter optim
            # TODO: remove numpy conversion
            updated_emulator.fit(x.numpy(), y.numpy())
        except Exception as e:
            print(f"Error refitting model: {e}")
            # If refitting fails, just return the original model
            return existing_emulator

        return updated_emulator

    def update_progress_bar(
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
        pbar.set_postfix(
            {
                "samples": n_samples,
                "failed": failed_count,
                "NROY": n_nroy_samples,
                "min_impl": f"{torch.min(impl_scores) if impl_scores.size(0) > 0 else 'NaN':.2f}",
                "max_impl": f"{torch.max(impl_scores) if impl_scores.size(0) > 0 else 'NaN':.2f}",
            }
        )
