from typing import Optional, Union

import numpy as np
from tqdm import tqdm

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
    ):
        """
        Initialize the history matcher.

        TODO in separate PR for #406:
        - make this work with experimental GP emulators
             - add check that provided emulator returns distribution (mean + variance)
        - refactor to use tensors instead of numpy arrays
        - make this work with updated Simulator class (after #414 is merged)

        Parameters
        ----------
            simulator: Simulator
                The simulation to emulate.
            TODO: could one ever pass multiple observations?
            observations: dict[str, tuple[float, float]]
                For each output variable, specifies observed [mean, variance] values.
            threshold: float
                Implausibility threshold (query points with implausability scores that
                exceed this value are ruled out). Defaults to 3, which is considered
                good value for simulations with a single output.
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
        # TODO: rank can't be more than output dimension, add a check
        self.rank = rank

        # save mean and variance of observations
        if not set(observations.keys()).issubset(set(self.simulator.output_names)):
            raise ValueError(
                f"Observation keys {set(observations.keys())} must be a subset of ",
                f"simulator output names {set(self.simulator.output_names)}",
            )
        obs_means = np.array(
            [observations[name][0] for name in self.simulator.output_names]
        )
        obs_vars = np.array(
            [observations[name][1] for name in self.simulator.output_names]
        )
        # Reshape observation arrays for broadcasting
        self.obs_means = obs_means.reshape(1, -1)  # [1, n_outputs]
        self.obs_vars = obs_vars.reshape(1, -1)  # [1, n_outputs]

    def calculate_implausibility(
        self,
        pred_means: np.ndarray,  # Shape [n_samples, n_outputs]
        pred_vars: np.ndarray,  # Shape [n_samples, n_outputs]
    ) -> dict[str, Union[np.ndarray, list[int]]]:
        """
        Calculate implausibility scores and identify NROY points.

        Parameters
        ----------
            pred_means: np.ndarray
                Array of prediction means [n_samples, n_outputs]
            pred_vars: np.ndarray
                Array of prediction variances [n_samples, n_outputs]

        Returns
        -------
            dict[str, union[np.ndarray, list[int]]]
                Contains the following key, value pairs:
                    - 'I': array of implausibility scores [n_samples, n_outputs]
                    - 'NROY': list of indices of Not Ruled Out Yet points
                    - 'RO': list of indices of Ruled Out points
        """
        # Add model discrepancy
        discrepancy = np.full_like(self.obs_vars, self.discrepancy)

        # Calculate total variance
        Vs = pred_vars + discrepancy + self.obs_vars

        # Calculate implausibility
        I = np.abs(self.obs_means - pred_means) / np.sqrt(Vs)

        # Determine NROY points based on rank parameter
        if self.rank == 1:
            # First-order implausibility: all outputs must satisfy threshold
            nroy_mask = np.all(I <= self.threshold, axis=1)
        else:
            # Sort implausibilities for each sample (descending)
            I_sorted = np.sort(I, axis=1)[:, ::-1]
            # The rank-th highest implausibility must be <= threshold
            nroy_mask = I_sorted[:, self.rank - 1] <= self.threshold

        # Get indices of NROY and RO points
        NROY = np.where(nroy_mask)[0]
        RO = np.where(~nroy_mask)[0]

        return {
            "I": I,  # Implausibility scores
            "NROY": list(NROY),  # Indices of NROY points
            "RO": list(RO),  # Indices of RO points
        }

    def sample_nroy(
        self,
        nroy_samples: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        TODO: update method to fix issues listed in #460
        TODO: we do random sampling here so need to fix random seed somewhere

        Generate new parameter samples within NROY space.

        Parameters
        ----------
            nroy_samples: np.ndarray
                Array of parameter samples in the NROY space [n_samples, n_parameters].
            n_samples: int
                Number of new samples to generate within the NROY bounds.

        Returns
        -------
            np.ndarray
                Array of parameter samples [n_samples, n_parameters].
        """

        # Sample uniformly within NROY bounds
        min_bounds = np.min(nroy_samples, axis=0)
        max_bounds = np.max(nroy_samples, axis=0)
        new_samples = np.random.uniform(
            min_bounds, max_bounds, size=(n_samples, nroy_samples.shape[1])
        )

        return new_samples

    def predict(
        self,
        X: np.ndarray,
        # TODO: update emulator object passed here
        emulator: Optional[object] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions for a batch of inputs X. Uses `self.simulator` unless
        an emulator trained on `self.simulator` data is provided.

        Parameters
        ----------
        X: np.ndarray
            Array of parameter samples to simulate/emulate [n_samples, n_parameters]
            returned by `self.simulator.sample_inputs` or `self.sample_nroy` methods.
        TODO: update emulator type and description below
        emulator: optional object
            Gaussian process emulator trained on self.simulator data.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Arrays of predicted means and variances as well as the input data for
            which predictions were made succesfully.
        """
        # TODO: when does this happen? do we need this?
        if X.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        # Make predictions using emulator
        if emulator is not None:
            pred_means, pred_stds = emulator.predict(X, return_std=True)
            pred_vars = pred_stds**2

            # TODO: don't need this once remove sklearn dependence
            # Ensure correct shape for single output case
            if len(pred_means.shape) == 1:
                pred_means = pred_means.reshape(-1, 1)
                pred_vars = pred_vars.reshape(-1, 1)

        # Make predictions using simulator
        else:
            results = self.simulator.run_batch_simulations(X)

            # Filter out failed simulations
            valid_indices = [i for i, x in enumerate(results) if x is not None]
            X = X[valid_indices]
            pred_means = results[valid_indices]
            pred_vars = np.full_like(pred_means, 0.01)  # Small fixed variance
            if pred_means.shape[0] == 0:
                # All simulations failed
                return np.array([]), np.array([]), np.array([])

        # Also return input vector in case simulation failed for some inputs
        return pred_means, pred_vars, X

    def run(
        self,
        n_waves: int = 3,
        n_samples_per_wave: int = 100,
        emulator_predict: bool = True,
        # TODO: update emulator type passed here
        initial_emulator: Optional[object] = None,
    ) -> tuple[np.ndarray, np.ndarray, Union[object, None]]:
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
        tuple[np.ndarray, np.ndarray, union[object, None]]
            - Array of all parameter samples for which predictions were made
            - Array of all implausability scores
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

        with tqdm(total=n_waves, desc="History Matching", unit="wave") as pbar:
            for wave in range(n_waves):
                # Run wave using batch processing
                pred_means, pred_vars, successful_samples = self.predict(
                    X=current_samples,
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
                if impl_scores.size > 0:
                    # Only include samples with scores
                    all_samples.append(successful_samples)
                    all_impl_scores.append(impl_scores)

                    # Update emulator if simulated data (succesfully)
                    if (not emulator_predict) and len(nroy_samples) > 10:
                        emulator = self.update_emulator(
                            emulator, successful_samples, pred_means
                        )

                # Generate new samples for next wave
                if wave < n_waves - 1:
                    if nroy_samples.size > 0:
                        current_samples = self.sample_nroy(
                            nroy_samples, n_samples_per_wave
                        )
                    else:
                        # If no NROY points, sample from full space
                        current_samples = self.simulator.sample_inputs(
                            n_samples_per_wave
                        )

                pbar.update(1)

        # Concatenate all samples and implausibility scores
        # TODO: should this include wave information in some form?
        final_samples = np.concatenate(all_samples) if all_samples else np.array([])

        final_impl_scores = (
            np.concatenate(all_impl_scores) if all_impl_scores else np.array([])
        )

        return final_samples, final_impl_scores, emulator

    def update_emulator(
        self,
        # TODO: update emulator type passed here
        existing_emulator: object,
        X: np.ndarray,
        y: np.ndarray,
        include_previous_data: bool = True,
    ):
        """
        Update an existing GP emulator with new training data.

        Parameters
        ----------
            TODO: eventually this should be autoemulate.GaussianProcessExact
            existing_emulator: Gaussian Process from sklearn
                Trained GP emulator.
            X: np.ndarray
                Array of parameter values to train emulator on.
            y: np.ndarray
                Array of output values.
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
        if (
            include_previous_data
            and hasattr(existing_emulator, "X_train_")
            and hasattr(existing_emulator, "y_train_")
        ):
            # Combine old and new training data
            X_combined = np.vstack((existing_emulator.X_train_, X))

            # Check if we're dealing with multi-output or single-output
            if len(existing_emulator.y_train_.shape) > 1 and len(y.shape) > 1:
                y_combined = np.vstack((existing_emulator.y_train_, y))
            else:
                y_combined = np.concatenate((existing_emulator.y_train_, y))
        else:
            # Just use new data
            X_combined = X
            y_combined = y

        # Update the emulator
        try:
            # Refit the entire model with new hyperparameters
            updated_emulator.fit(X_combined, y_combined)
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
            total_samples - len(impl_scores) if impl_scores.size > 0 else total_samples
        )
        pbar.set_postfix(
            {
                "samples": len(impl_scores),
                "failed": failed_count,
                "NROY": len(nroy_samples),
                "min_impl": f"{np.min(impl_scores) if impl_scores.size > 0 else 'NaN':.2f}",
                "max_impl": f"{np.max(impl_scores) if impl_scores.size > 0 else 'NaN':.2f}",
            }
        )
