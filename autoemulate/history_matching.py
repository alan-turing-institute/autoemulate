import sys
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from tqdm import tqdm

from autoemulate.simulations.base import Simulator


class HistoryMatcher:
    def __init__(
        self,
        simulator: Simulator,
        observations: Dict[str, Tuple[float, float]],
        threshold: float = 3.0,
        model_discrepancy: float = 0.0,
        rank: int = 1,
    ):
        """
        Initialize the history matcher

        Args:
            simulator: Simulator instance that implements BaseSimulator
            observations: Dictionary mapping output names to (mean, variance) pairs
            threshold: Implausibility threshold
            model_discrepancy: Model discrepancy term
            rank: Rank for history matching
        """
        self.simulator = simulator
        self.observations = observations
        self.threshold = threshold
        self.discrepancy = model_discrepancy
        self.rank = rank

        # In your code, modify HistoryMatcher.__init__ to change this line:
        if not set(self.observations.keys()).issubset(set(self.simulator.output_names)):
            raise ValueError(
                f"Observation keys {set(self.observations.keys())} must be a subset of simulator output names {set(self.simulator.output_names)}"
            )

    def calculate_implausibility(
        self, predictions: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Calculate implausibility for given predictions

        Args:
            predictions: Dictionary mapping output names to (mean, variance) pairs

        Returns:
            Dictionary with:
            - 'I': array of implausibility scores
            - 'NROY': indices of Not Ruled Out Yet points
            - 'RO': indices of Ruled Out points
        """
        obs_means = np.array(
            [self.observations[name][0] for name in self.simulator.output_names]
        )
        obs_vars = np.array(
            [self.observations[name][1] for name in self.simulator.output_names]
        )
        pred_means = np.array(
            [predictions[name][0] for name in self.simulator.output_names]
        )
        pred_vars = np.array(
            [predictions[name][1] for name in self.simulator.output_names]
        )

        if len(obs_means) != len(pred_means):
            raise ValueError("Mismatch between observations and predictions")

        discrepancy = np.full(len(obs_means), self.discrepancy)
        Vs = pred_vars + discrepancy + obs_vars
        I = np.abs(obs_means - pred_means) / np.sqrt(Vs)

        NROY = np.where(I <= self.threshold)[0]
        RO = np.where(I > self.threshold)[0]

        return {"I": I, "NROY": list(NROY), "RO": list(RO)}

    def run_wave(
        self,
        parameter_samples: List[Dict[str, float]],
        use_emulator: bool = False,
        emulator: Optional[object] = None,
    ) -> Tuple[List[Dict[str, float]], np.ndarray]:
        """
        Run a wave of simulations or emulator predictions

        Args:
            parameter_samples: List of parameter dictionaries to evaluate
            use_emulator: Whether to use emulator instead of simulations
            emulator: Trained emulator object (if use_emulator is True)

        Returns:
            Tuple of (successful_samples, impl_scores)
        """
        successful_samples = []
        impl_scores = []

        for params in parameter_samples:
            if use_emulator and emulator is not None:
                # Use emulator for predictions
                X = np.array(
                    [params[name] for name in self.simulator.param_names]
                ).reshape(1, -1)

                # Handle both single-output and multi-output cases
                pred_means, pred_stds = emulator.predict(X, return_std=True)
                pred_vars = pred_stds**2

                # Check if we're dealing with 1D arrays (single output) or 2D arrays (multi-output)
                if len(pred_means.shape) == 1:
                    # Single output case - each output has its own emulator
                    predictions = {
                        name: (pred_means[i], pred_vars[i])
                        for i, name in enumerate(self.simulator.output_names)
                    }
                else:
                    # Multi-output case - one emulator for all outputs
                    predictions = {
                        name: (pred_means[0, i], pred_vars[0, i])
                        for i, name in enumerate(self.simulator.output_names)
                    }
            else:
                # Run actual simulation
                outputs = self.simulator.sample_forward(params)
                if outputs is None:
                    continue

                # Create predictions dictionary
                predictions = {
                    name: (outputs[i], 0.01)  # Small fixed variance
                    for i, name in enumerate(self.simulator.output_names)
                }

            # Calculate implausibility
            result = self.calculate_implausibility(predictions)
            impl_scores.append(result["I"])
            successful_samples.append(params)

        return successful_samples, np.array(impl_scores)

    def generate_new_samples(
        self, nroy_samples: List[Dict[str, float]], n_samples: int
    ) -> List[Dict[str, float]]:
        """
        Generate new parameter samples within NROY space

        Args:
            nroy_samples: List of NROY parameter sets
            n_samples: Number of new samples to generate

        Returns:
            List of new parameter dictionaries
        """
        if not nroy_samples:
            return self.simulator.sample_inputs(n_samples)

        # Convert to array
        X_nroy = np.array(
            [
                [sample[name] for name in self.simulator.param_names]
                for sample in nroy_samples
            ]
        )

        # Sample uniformly within NROY bounds
        min_bounds = np.min(X_nroy, axis=0)
        max_bounds = np.max(X_nroy, axis=0)
        new_samples = np.random.uniform(
            min_bounds, max_bounds, size=(n_samples, X_nroy.shape[1])
        )

        # Convert back to dictionaries
        return [dict(zip(self.simulator.param_names, sample)) for sample in new_samples]

    def run_history_matching(
        self,
        n_waves: int = 3,
        n_samples_per_wave: int = 100,
        use_emulator: bool = True,  # Set to True by default
        initial_emulator=None,  # Add parameter for your pre-trained GP
    ):
        """
        Run iterative history matching with an existing emulator

        Args:
            n_waves: Number of waves to run
            n_samples_per_wave: Number of samples per wave
            use_emulator: Whether to use emulator
            initial_emulator: Pre-trained emulator/GP to start with

        Returns:
            Tuple of (all_samples, all_impl_scores, updated_emulator)
        """
        all_samples = []
        all_impl_scores = []
        emulator = initial_emulator  # Start with your pre-trained GP

        # Generate initial samples
        current_samples = self.simulator.sample_inputs(n_samples_per_wave)

        with tqdm(total=n_waves, desc="History Matching", unit="wave") as pbar:
            for wave in range(n_waves):
                # Determine if we should use emulator
                wave_use_emulator = use_emulator and (emulator is not None)

                # Run the wave
                successful_samples, impl_scores = self.run_wave(
                    current_samples, use_emulator=wave_use_emulator, emulator=emulator
                )

                # Handle case where no simulations were successful
                if len(successful_samples) == 0:
                    tqdm.write("Warning: No successful simulations in this wave")
                    if wave < n_waves - 1:
                        current_samples = self.simulator.sample_inputs(
                            n_samples_per_wave
                        )
                    pbar.update(1)
                    continue

                # Store results
                all_samples.extend(successful_samples)
                all_impl_scores.extend(impl_scores)

                for sample in successful_samples:
                    # Add wave information to each sample
                    sample["wave"] = wave + 1  # 1-indexed waves

                # Update progress bar with informative statistics
                if len(impl_scores) > 0:
                    nroy_mask = np.all(impl_scores <= self.threshold, axis=1)
                    nroy_samples = [
                        s for s, m in zip(successful_samples, nroy_mask) if m
                    ]

                    pbar.set_postfix(
                        samples=len(successful_samples),
                        min_impl=f"{np.min(impl_scores):.2f}",
                        max_impl=f"{np.max(impl_scores):.2f}",
                        nroy=len(nroy_samples),
                    )
                else:
                    pbar.set_postfix(samples=len(successful_samples), nroy=0)

                # After each wave, update your GP with new samples
                if len(successful_samples) > 10:
                    tqdm.write("Updating emulator with new samples...")
                    # Get outputs from simulator to train the emulator
                    X_train = np.array(
                        [
                            [sample[name] for name in self.simulator.param_names]
                            for sample in successful_samples
                        ]
                    )

                    # Get the simulator outputs (not implausibility scores)
                    y_train = np.array(
                        [
                            [
                                self.simulator.sample_forward(params)[i]
                                for i in range(len(self.simulator.output_names))
                            ]
                            for params in successful_samples
                        ]
                    )

                    # Update the emulator with proper input-output pairs
                    emulator = self.update_emulator(emulator, X_train, y_train)

                # Generate new samples for next wave if not last wave
                if wave < n_waves - 1:
                    if len(nroy_samples) > 0:
                        current_samples = self.generate_new_samples(
                            nroy_samples, n_samples_per_wave
                        )
                    else:
                        # Include this information in the postfix instead of printing it
                        pbar.set_postfix(
                            samples=len(successful_samples),
                            min_impl=f"{np.min(impl_scores) if len(impl_scores) > 0 else 'N/A'}",
                            max_impl=f"{np.max(impl_scores) if len(impl_scores) > 0 else 'N/A'}",
                            nroy=0,
                            status="generating new samples",
                        )
                        current_samples = self.simulator.sample_inputs(
                            n_samples_per_wave
                        )

                # Update progress bar
                pbar.update(1)

        # Return the updated emulator along with other results
        return all_samples, np.array(all_impl_scores), emulator

    def update_emulator(
        self,
        existing_emulator,
        new_samples,
        new_outputs,
        include_previous_data=True,
        refit_hyperparams=False,
    ):
        """
        Update an existing GP emulator with new training data.

        Args:
            existing_emulator: Trained GP emulator (assumes sklearn.gaussian_process or similar API)
            new_samples: List of dictionaries with parameter values or numpy array
            new_outputs: Array of corresponding output values
            include_previous_data: Whether to include previous training data (default: True)
            refit_hyperparams: Whether to refit hyperparameters on the combined dataset (default: False)

        Returns:
            Updated GP emulator
        """
        # Instead of deepcopy, we'll create a new instance if needed
        # For now, just use the existing model as is
        updated_emulator = existing_emulator

        # Convert new_samples to numpy array if it's a list of dictionaries
        if isinstance(new_samples[0], dict):
            # Extract parameter names from the first dictionary
            param_names = list(new_samples[0].keys())
            # Convert to numpy array
            X_new = np.array(
                [[sample[name] for name in param_names] for sample in new_samples]
            )
        else:
            # Already a numpy array
            X_new = np.array(new_samples)

        # Convert new_outputs to numpy array if needed
        y_new = np.array(new_outputs)

        # If we need to include previous data and emulator has stored training data
        if (
            include_previous_data
            and hasattr(existing_emulator, "X_train_")
            and hasattr(existing_emulator, "y_train_")
        ):
            # Combine old and new training data
            X_combined = np.vstack((existing_emulator.X_train_, X_new))

            # Check if we're dealing with multi-output or single-output
            if len(existing_emulator.y_train_.shape) > 1 and len(y_new.shape) > 1:
                y_combined = np.vstack((existing_emulator.y_train_, y_new))
            else:
                y_combined = np.concatenate((existing_emulator.y_train_, y_new))
        else:
            # Just use new data
            X_combined = X_new
            y_combined = y_new

        # Update the emulator
        if refit_hyperparams:
            try:
                # Refit the entire model with new hyperparameters
                updated_emulator.fit(X_combined, y_combined)
            except Exception as e:
                print(f"Error refitting model: {e}")
                # If refitting fails, just return the original model
                return existing_emulator
        else:
            # For models without an explicit update method, we'll try a simpler approach
            try:
                # Store the training data for future reference
                updated_emulator.X_train_ = X_combined
                updated_emulator.y_train_ = y_combined

                # Some models might have methods like update or partial_fit
                if hasattr(updated_emulator, "update") and callable(
                    getattr(updated_emulator, "update")
                ):
                    updated_emulator.update(X_new, y_new)
                elif hasattr(updated_emulator, "partial_fit") and callable(
                    getattr(updated_emulator, "partial_fit")
                ):
                    updated_emulator.partial_fit(X_new, y_new)
                else:
                    # As a fallback, attempt to refit
                    updated_emulator.fit(X_combined, y_combined)
            except Exception as e:
                print(f"Error updating model: {e}")
                # If updating fails, just return the original model
                return existing_emulator

        return updated_emulator
