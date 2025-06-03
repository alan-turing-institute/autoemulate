from typing import Optional
from typing import Union

import torch
from tqdm import tqdm

from autoemulate.experimental.types import TensorLike
from autoemulate.simulations.base import Simulator


class HistoryMatching:
    """
    History matching is a model calibration method, which uses observed data to rule out
    parameter values which are ``implausible``. The implausability metric is:

    .. math::
        I_i(\bar{x_0}) = \frac{|z_i - \mathbb{E}(f_i(\bar{x_0}))|}
        {\sqrt{\text{Var}[z_i - \mathbb{E}(f_i(\bar{x_0}))]}}

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
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the history matcher.

        TODO in separate PR for #406:
        - make this work with experimental GP emulators
        - refactor to use tensors instead of numpy arrays
        - make this work with updated Simulator class (after #414 is merged)
        - check whether should handle device throughout

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
            random_seed: optional int
                Random seed to set for reproducibility of sampling.
        """
        self.simulator = simulator
        self.threshold = threshold
        self.discrepancy = model_discrepancy
        # TODO: rank can't be more than output dimension, add a check
        self.rank = rank
        # TODO: handle random seed (only used when sampling)

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

    def calculate_implausibility(
        self,
        pred_means: TensorLike,  # Shape [n_samples, n_outputs]
        pred_vars: Optional[TensorLike] = None,  # Shape [n_samples, n_outputs]
        default_var: float = 0.01,
    ) -> dict[str, Union[TensorLike, list[int]]]:
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
            dict[str, union[TensorLike, list[int]]]
                Contains the following key, value pairs:
                    - 'I': tensor of implausibility scores [n_samples, n_outputs]
                    - 'NROY': list of indices of Not Ruled Out Yet points
                    - 'RO': list of indices of Ruled Out points
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
            nroy_mask = torch.all(I <= self.threshold, dim=1)
        else:
            # Sort implausibilities for each sample (descending)
            I_sorted = torch.sort(I, dim=1, descending=True)
            # The rank-th highest implausibility must be <= threshold
            nroy_mask = I_sorted[:, self.rank - 1] <= self.threshold

        # Get indices of NROY and RO samples
        NROY = torch.where(nroy_mask)[0]
        RO = torch.where(~nroy_mask)[0]

        return {
            "I": I,  # Implausibility scores
            # TODO: do we need to turn these into lists?
            "NROY": NROY.tolist(),  # Indices of NROY points
            "RO": RO.tolist(),  # Indices of RO points
        }

    def sample_nroy(
        self,
        nroy_samples: TensorLike,
        n_samples: int,
    ) -> TensorLike:
        """
        Generate new parameter samples within NROY space.

        Parameters
        ----------
            nroy_samples: TensorLike
                Tensor of parameter samples in the NROY space [n_samples, n_parameters].
            n_samples: int
                Number of new samples to generate within the NROY bounds.

        Returns
        -------
            TensorLike
                Tensor of parameter samples [n_samples, n_parameters].
        """

        min_bounds, _ = torch.min(nroy_samples, dim=0)
        max_bounds, _ = torch.max(nroy_samples, dim=0)

        # Need to handle discontinuous NROY spaces
        # i.e., a region within min/max bounds is RO
        valid_samples = torch.empty(
            (0, nroy_samples.shape[1]),
            dtype=nroy_samples.dtype,
            device=nroy_samples.device,
        )
        while len(valid_samples) < n_samples:
            # Generate candidates
            candidate_samples = (
                torch.rand((n_samples, nroy_samples.shape[1]))
                * (max_bounds - min_bounds)
                + min_bounds
            )

            # Filter valid samples based on implausibility and concatenate
            implausibility = self.calculate_implausibility(candidate_samples)
            valid_candidates = candidate_samples[implausibility["NROY"]]
            valid_samples = torch.cat((valid_samples, valid_candidates), dim=0)

            # Only return required number of samples
            if len(valid_samples) > n_samples:
                valid_samples = valid_samples[:n_samples]

        return valid_samples

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
            Tensor of parameter samples to simulate/emulate [n_samples, n_parameters]
            returned by `self.simulator.sample_inputs` or `self.sample_nroy` methods.
        TODO: update emulator type and description below
        emulator: optional object
            Gaussian process emulator trained on self.simulator data.

        Returns
        -------
        tuple[TensorLike, TensorLike, TensorLike]
            Arrays of predicted means and variances as well as the input data for
            which predictions were made succesfully.
        """
        if x.shape[0] == 0:
            # TODO: use torch.empty here?
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
        TODO: can we simplify this?
        tuple[TensorLike, TensorLike, union[object, None]]
            - Tensor of all parameter samples for which predictions were made
            - Tensor of all implausability scores
            - a GP emulator (retrained on new data if `emulator_predict=False`) or None
        """
        if emulator_predict and initial_emulator is None:
            raise ValueError(
                "Need to pass a GP emulator object when `emulator_predict=True`"
            )

        all_samples = []
        all_impl_scores = []
        emulator = initial_emulator
        current_samples = self.simulator.sample_inputs(n_samples_per_wave)
        current_samples = torch.from_numpy(current_samples)

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
                    pbar, current_samples, impl_scores, nroy_samples
                )

                # Store results
                if impl_scores.size(0) > 0:
                    # Only include samples with scores
                    all_samples.append(successful_samples)
                    all_impl_scores.append(impl_scores)

                    # Update emulator if simulated (enough) data
                    if (not emulator_predict) and len(nroy_samples) > 10:
                        emulator = self.update_emulator(
                            emulator, successful_samples, pred_means
                        )

                # Generate new samples for next wave
                if wave < n_waves - 1:
                    if nroy_samples.size(0) > 0:
                        current_samples = self.sample_nroy(
                            nroy_samples, n_samples_per_wave
                        )
                    else:
                        # If no NROY points, sample from full space
                        current_samples = self.simulator.sample_inputs(
                            n_samples_per_wave
                        )
                        current_samples = torch.from_numpy(current_samples)

                pbar.update(1)

        # Concatenate all samples and implausibility scores
        # TODO: should this include wave information?
        final_samples = torch.cat(all_samples) if all_samples else torch.Tensor([])

        final_impl_scores = (
            torch.cat(all_impl_scores) if all_impl_scores else torch.Tensor([])
        )

        return final_samples, final_impl_scores, emulator

    def update_emulator(
        self,
        # TODO: update emulator type passed here
        existing_emulator: object,
        x: TensorLike,
        y: TensorLike,
        include_previous_data: bool = True,
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
            include_previous_data: bool
                Whether to include previous training data (default: True)

        Returns
        -------
            Updated GP emulator
        """
        # Instead of deepcopy, we'll create a new instance if needed
        # For now, just use the existing model as is
        updated_emulator = existing_emulator

        # If we need to include previous data and emulator has stored training data
        # TODO: should data be stored in HistoryMatcher (same as ActiveLearner)?
        # that way can make sure we keep appending data to it on each retrain
        if (
            include_previous_data
            and hasattr(existing_emulator, "X_train_")
            and hasattr(existing_emulator, "y_train_")
        ):
            # Combine old and new training data
            # TODO: remove numpy conversion
            X_combined = torch.cat(
                (torch.from_numpy(existing_emulator.X_train_), x), dim=0
            )
            y_combined = torch.cat(
                (torch.from_numpy(existing_emulator.y_train_), y), dim=0
            )
        else:
            # Just use new data
            X_combined = x
            y_combined = y

        # Update the emulator
        try:
            # Refit the entire model, includes hyperparameter optim
            # TODO: remove numpy conversion
            updated_emulator.fit(X_combined.numpy(), y_combined.numpy())
        except Exception as e:
            print(f"Error refitting model: {e}")
            # If refitting fails, just return the original model
            return existing_emulator

        return updated_emulator

    def update_progress_bar(self, pbar, current_samples, impl_scores, nroy_samples):
        """
        Updates the progress bar.
        """
        total_samples = len(current_samples)
        failed_count = (
            total_samples - len(impl_scores)
            if impl_scores.size(0) > 0
            else total_samples
        )
        # TODO: check if min/max works correctly here (i.e., what is dim of impl_scores)
        pbar.set_postfix(
            {
                "samples": len(impl_scores),
                "failed": failed_count,
                "NROY": len(nroy_samples),
                "min_impl": f"{torch.min(impl_scores) if impl_scores.size(0) > 0 else 'NaN':.2f}",
                "max_impl": f"{torch.max(impl_scores) if impl_scores.size(0) > 0 else 'NaN':.2f}",
            }
        )
