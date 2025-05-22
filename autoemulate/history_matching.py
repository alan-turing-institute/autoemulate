from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
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
             - add check that provided emulator returns distribution (need mean + variance)
        - refactor to use torch instead of numpy
        - make this work with updated Simulator class (after #414 is merged)

        Parameters
        ----------
            simulator: Simulator
                The simulation to emulate.
            observations: dict[str, tuple[float, float]]
                Maps output names to (mean, variance) pairs.
            threshold: float
                Implausibility threshold (query points with implausability scores that
                exceed this value are ruled out).
            model_discrepancy: float
                Additional variance to include in the implausability calculation.
            rank: int
                TODO is the below correct? adapted from mogp_emulator docs:
                NOTE that mogp_emulator has a different default (2nd largest score)
                Scoring method for multiple outputs. Must be a non-negative
                integer less than the number of observations, which denotes
                the location in the rank ordering of implausibility values
                where the score is evaluated (i.e. the default value of ``1``
                indicates that the largest implausibility will be used).
        """
        self.simulator = simulator
        self.threshold = threshold
        self.discrepancy = model_discrepancy
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
            # Higher-order implausibility:
            # - the nth highest implausibility must satisfy threshold
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

    def run_wave(
        self,
        X: np.ndarray,
        # TODO: update emulator object passed here
        emulator: Optional[object] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run a wave of simulations or emulator predictions with batch support.
        Using simulator to make predictions unless emulator object is passed.

        Parameters
        ----------
        X: np.ndarray
            Array of parameter samples to simulate/emulate [n_samples, n_parameters]
            returned by `self.simulator.sample_inputs` method.
        TODO: update emulator type and description below
        emulator: optional object
            Gaussian process emulator to use to make predictions.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Arrays of NROY parameters and of all implausibility scores.
        """
        # TODO: check when does this happen? do we need this?
        if X.shape[0] == 0:
            return np.array([]), np.array([])

        # Make predictions using emulator
        if emulator is not None:
            pred_means, pred_stds = emulator.predict(X, return_std=True)
            pred_vars = pred_stds**2

            # Ensure correct shape for single output case
            if len(pred_means.shape) == 1:
                pred_means = pred_means.reshape(-1, 1)
                pred_vars = pred_vars.reshape(-1, 1)
        # Make predictions using simulator
        # Requires checking if simulation failed
        else:
            # Returns array of output stats [n_samples, n_stats]
            results = self.simulator.run_batch_simulations(X)

            # Filter out failed simulations
            valid_indices = [i for i, x in enumerate(results) if x is not None]
            X = X[valid_indices]
            pred_means = results[valid_indices]
            pred_vars = np.full_like(pred_means, 0.01)  # Small fixed variance
            if pred_means.shape[0] == 0:
                # All simulations failed
                return np.array([]), np.array([])

        # Calculate implausibility in batch
        implausibility = self.calculate_implausibility(pred_means, pred_vars)

        # NROY parameters and all parameter implausibility scores
        return X[implausibility["NROY"]], implausibility["I"]

    def run(
        self,
        n_waves: int = 3,
        n_samples_per_wave: int = 100,
        # TODO: update emulator type passed here
        initial_emulator: Optional[object] = None,
    ):
        # TODO: add return type
        """
        Run iterative history matching. In each wave:
            - sample parameter values to test
            - make predictions for the parameter samples
                - if no `initial_emulator` is passed, use simulator to make
                  predictions in the first wave, otherwise use emulator
            - compute implausability scores
            - (re)train emulator using all data

        Parameters
        ----------
        n_waves: int
            Number of iterations of history matching to run.
        n_samples_per_wave: int
            Number of parameter samples to make predictions for in each wave.
        TODO: update emulator type and description below
        initial_emulator: optional object
            A pre-trained Gaussian process emulator to use to make predictions.
            If not passed then an emulator is trained during the first wave. In
            all consecutive waves, the passed or created enmulator is updated.

        Returns
        -------
        TODO
        """
        all_samples = []
        all_impl_scores = []
        emulator = initial_emulator
        current_samples = self.simulator.sample_inputs(n_samples_per_wave)

        with tqdm(total=n_waves, desc="History Matching", unit="wave") as pbar:
            for wave in range(n_waves):
                # Run wave using batch processing
                successful_samples, impl_scores = self.run_wave(
                    X=current_samples,
                    emulator=emulator,
                )

                # Update tracking metrics
                nroy_count = len(successful_samples)
                total_samples = len(current_samples)
                failed_count = (
                    total_samples - len(impl_scores)
                    if impl_scores.size > 0
                    else total_samples
                )

                # Update progress bar
                pbar.set_postfix(
                    {
                        "samples": len(impl_scores) if impl_scores.size > 0 else 0,
                        "failed": failed_count,
                        "NROY": nroy_count,
                        "min_impl": f"{np.min(impl_scores) if impl_scores.size > 0 else 'NaN':.2f}",
                        "max_impl": f"{np.max(impl_scores) if impl_scores.size > 0 else 'NaN':.2f}",
                    }
                )

                # Store results
                if impl_scores.size > 0:
                    all_samples.extend(
                        [
                            {**params, "wave": wave + 1}
                            for params in current_samples[
                                : len(impl_scores)
                            ]  # Only include samples with scores
                        ]
                    )
                    all_impl_scores.append(impl_scores)

                    # Update emulator if not using emulator in this wave
                    # TODO: shouldn't this always be updated, not just if
                    # no emulator was available to begin with?
                    if (emulator is None) and len(successful_samples) > 10:
                        X_train = successful_samples
                        y_train = self.simulator.run_batch_simulations(
                            successful_samples
                        )
                        if len(y_train) > 0:
                            emulator = self.update_emulator(emulator, X_train, y_train)

                # Generate new samples for next wave
                if wave < n_waves - 1:
                    if successful_samples:
                        current_samples = self.sample_nroy(
                            successful_samples, n_samples_per_wave
                        )
                    else:
                        # If no NROY points, sample from full space
                        current_samples = self.simulator.sample_inputs(
                            n_samples_per_wave
                        )

                pbar.update(1)

        # Concatenate all implausibility scores
        final_impl_scores = (
            np.concatenate(all_impl_scores) if all_impl_scores else np.array([])
        )

        # Q: what should all samples be returned as ?
        return all_samples, final_impl_scores, emulator

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
        if (
            include_previous_data
            and hasattr(existing_emulator, "X_train_")
            and hasattr(existing_emulator, "y_train_")
        ):
            # Combine old and new training data
            X_combined = np.vstack((existing_emulator.X_train_, X))

            # Check if we're dealing with multi-output or single-output
            # TODO: can we always just use vstack here?
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
