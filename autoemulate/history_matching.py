import sys
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import pandas as pd
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
        self, 
        pred_means: np.ndarray,  # Shape [n_samples, n_outputs]
        pred_vars: np.ndarray,   # Shape [n_samples, n_outputs]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate implausibility and identify NROY points
        
        Args:
            pred_means: Array of prediction means [n_samples, n_outputs]
            pred_vars: Array of prediction variances [n_samples, n_outputs]

        Returns:
            Dictionary with:
            - 'I': array of implausibility scores [n_samples, n_outputs]
            - 'NROY': indices of Not Ruled Out Yet points
            - 'RO': indices of Ruled Out points
        """
        # Get observation means and variances
        obs_means = np.array([self.observations[name][0] for name in self.simulator.output_names])
        obs_vars = np.array([self.observations[name][1] for name in self.simulator.output_names])
        
        # Add model discrepancy
        discrepancy = np.full_like(obs_vars, self.discrepancy)
        
        # Reshape observation arrays for broadcasting
        obs_means = obs_means.reshape(1, -1)  # [1, n_outputs]
        obs_vars = obs_vars.reshape(1, -1)    # [1, n_outputs]
        discrepancy = discrepancy.reshape(1, -1)  # [1, n_outputs]
        
        # Calculate total variance
        Vs = pred_vars + discrepancy + obs_vars
        
        # Calculate implausibility
        I = np.abs(obs_means - pred_means) / np.sqrt(Vs)
        
        # Determine NROY points based on rank parameter
        if self.rank == 1:
            # First-order implausibility: all outputs must satisfy threshold
            nroy_mask = np.all(I <= self.threshold, axis=1)
        else:
            # Higher-order implausibility: the nth highest implausibility must satisfy threshold
            # Sort implausibilities for each sample (descending)
            I_sorted = np.sort(I, axis=1)[:, ::-1]
            # The rank-th highest implausibility must be <= threshold
            nroy_mask = I_sorted[:, self.rank-1] <= self.threshold
        
        # Get indices of NROY and RO points
        NROY = np.where(nroy_mask)[0]
        RO = np.where(~nroy_mask)[0]
        
        return {
            "I": I,               # Implausibility scores
            "NROY": list(NROY),   # Indices of NROY points
            "RO": list(RO)        # Indices of RO points
        }


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
        
    def run_wave(
        self,
        parameter_samples: List[Dict[str, float]],
        use_emulator: bool = False,
        emulator: Optional[object] = None,
    ) -> Tuple[List[Dict[str, float]], np.ndarray]:
        """Run a wave of simulations or emulator predictions with batch support."""
        if not parameter_samples:
            return [], np.array([])

        # Convert samples to array format for batch processing
        X = np.array([
            [params[name] for name in self.simulator.param_names]
            for params in parameter_samples
        ])
        
        if use_emulator and emulator is not None:
            pred_means, pred_stds = emulator.predict(X, return_std=True)
            pred_vars = pred_stds**2
            
            # Ensure correct shape for single output case
            if len(pred_means.shape) == 1:
                pred_means = pred_means.reshape(-1, 1)
                pred_vars = pred_vars.reshape(-1, 1)

        else:

            sample_df = pd.DataFrame(parameter_samples)
            results = self.simulator.run_batch_simulations(sample_df)
            
            # Filter out failed simulations
            valid_indices = [i for i, x in enumerate(results) if x is not None]
            valid_samples = [parameter_samples[i] for i in valid_indices]
            valid_results = [results[i] for i in valid_indices]
            
            if not valid_results:
                return [], np.array([])
                
            pred_means = np.array(valid_results)
            pred_vars = np.full_like(pred_means, 0.01)  # Small fixed variance
            
            # Update X to only include valid samples
            X = np.array([
                [params[name] for name in self.simulator.param_names]
                for params in valid_samples
            ])
            
            parameter_samples = valid_samples

        # Calculate implausibility in batch
        implausibility = self.calculate_implausibility(pred_means, pred_vars)
        
        # Get NROY samples
        nroy_samples = [parameter_samples[i] for i in implausibility["NROY"]]
        
        # Get all implausibility scores
        all_impl_scores = implausibility["I"]
        
        return nroy_samples, all_impl_scores

    def run_history_matching(
        self,
        n_waves: int = 3,
        n_samples_per_wave: int = 100,
        use_emulator: bool = True,
        initial_emulator=None,
    ):
        """Run iterative history matching using the updated implausibility calculation."""
        all_samples = []
        all_impl_scores = []
        emulator = initial_emulator
        current_samples = self.simulator.sample_inputs(n_samples_per_wave)

        with tqdm(total=n_waves, desc="History Matching", unit="wave") as pbar:
            for wave in range(n_waves):
                wave_use_emulator = use_emulator and (emulator is not None)
                
                # Run wave using batch processing
                successful_samples, impl_scores = self.run_wave(
                    parameter_samples=current_samples,
                    use_emulator=wave_use_emulator,
                    emulator=emulator
                )
                
                # Update tracking metrics
                nroy_count = len(successful_samples)
                total_samples = len(current_samples)
                failed_count = total_samples - len(impl_scores) if impl_scores.size > 0 else total_samples
                
                # Update progress bar
                pbar.set_postfix({
                    "samples": len(impl_scores) if impl_scores.size > 0 else 0,
                    "failed": failed_count,
                    "NROY": nroy_count,
                    "min_impl": f"{np.min(impl_scores) if impl_scores.size > 0 else 'NaN':.2f}",
                    "max_impl": f"{np.max(impl_scores) if impl_scores.size > 0 else 'NaN':.2f}",
                })

                # Store results
                if impl_scores.size > 0:
                    all_samples.extend([
                        {**params, "wave": wave + 1}
                        for params in current_samples[:len(impl_scores)]  # Only include samples with scores
                    ])
                    all_impl_scores.append(impl_scores)

                    # Update emulator if not using emulator in this wave
                    if not wave_use_emulator and len(successful_samples) > 10:
                        X_train = np.array([
                            [s[name] for name in self.simulator.param_names]
                            for s in successful_samples
                        ])
                        y_train = np.array([
                            self.simulator.sample_forward(s)
                            for s in successful_samples
                        ])
                        
                        if len(y_train) > 0:
                            emulator = self.update_emulator(emulator, X_train, y_train)

                # Generate new samples for next wave
                if wave < n_waves - 1:
                    if successful_samples:
                        current_samples = self.generate_new_samples(
                            successful_samples, 
                            n_samples_per_wave
                        )
                    else:
                        # If no NROY points, sample from full space
                        current_samples = self.simulator.sample_inputs(n_samples_per_wave)

                pbar.update(1)

        # Concatenate all implausibility scores
        final_impl_scores = np.concatenate(all_impl_scores) if all_impl_scores else np.array([])
        
        return all_samples, final_impl_scores, emulator   


    def update_emulator(
        self,
        existing_emulator,
        new_samples,
        new_outputs,
        include_previous_data=True,
    ):
        """
        Update an existing GP emulator with new training data.

        Args:
            existing_emulator: Trained GP emulator (assumes sklearn.gaussian_process or similar API)
            new_samples: List of dictionaries with parameter values or numpy array
            new_outputs: Array of corresponding output values
            include_previous_data: Whether to include previous training data (default: True)

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
        try:
            # Refit the entire model with new hyperparameters
            updated_emulator.fit(X_combined, y_combined)
        except Exception as e:
            print(f"Error refitting model: {e}")
            # If refitting fails, just return the original model
            return existing_emulator
            
        return updated_emulator
