from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC
from pyro.infer import NUTS


class MCMCCalibrator:
    def __init__(
        self,
        emulator,
        sensitivity_results: pd.DataFrame,
        observations: Dict[str, Tuple[float, float]],
        parameter_bounds: Dict[str, List[float]],
        nroy_samples: Optional[np.ndarray] = None,
        nroy_indices: Optional[List[int]] = None,
        all_samples: Optional[np.ndarray] = None,
        top_n_params: int = 5,
        device: str = "cpu",
    ):
        self.emulator = emulator
        self.observations = observations
        self.device = device
        self.parameter_bounds = parameter_bounds
        self.nroy_samples = nroy_samples
        self.nroy_indices = nroy_indices
        self.all_samples = all_samples

        # Get important parameters from SA and set up bounds
        self.important_params = self._get_important_parameters(
            sensitivity_results, top_n_params
        )

        self.param_names = list(parameter_bounds.keys())
        self.important_param_indices = [
            self.param_names.index(p) for p in self.important_params
        ]

        # Set up parameter bounds (refined by NROY if available)
        self.calibration_bounds = self._setup_calibration_bounds()

        print(f"Calibrating top {len(self.important_params)} parameters:")
        print("\n".join(f"  - {param}" for param in self.important_params))

    def _get_important_parameters(
        self, sensitivity_results: pd.DataFrame, top_n: int
    ) -> List[str]:
        """Get top parameters based on total Sobol indices."""
        st_results = sensitivity_results[sensitivity_results["index"] == "ST"]
        return (
            st_results.groupby("parameter")["value"]
            .mean()
            .nlargest(top_n)
            .index.tolist()
        )

    def _setup_calibration_bounds(self) -> Dict[str, List[float]]:
        """Set up calibration bounds, refining with NROY samples if available."""
        if not self._has_valid_nroy_samples():
            print("No valid NROY samples. Using original parameter bounds.")
            return {p: self.parameter_bounds[p] for p in self.important_params}

        print(f"Using {len(self.nroy_samples)} NROY samples to refine bounds")
        return self._refine_bounds_from_nroy()

    def _has_valid_nroy_samples(self) -> bool:
        """Check if NROY samples are valid and usable."""
        if self.nroy_samples is None:
            return False
        if len(self.nroy_samples) == 0:
            print("Warning: Empty NROY samples provided.")
            return False
        if self.nroy_samples.shape[1] != len(self.parameter_bounds):
            print(
                f"Warning: NROY samples has {self.nroy_samples.shape[1]} parameters, "
                f"but expected {len(self.parameter_bounds)}."
            )
            return False
        return True

    def _refine_bounds_from_nroy(self) -> Dict[str, List[float]]:
        """Refine parameter bounds using NROY samples."""
        refined_bounds = {}

        for i, param_name in enumerate(self.param_names):
            if param_name not in self.important_params:
                continue

            try:
                nroy_values = self.nroy_samples[:, i]
                if len(nroy_values) == 0:
                    raise ValueError("No NROY values")

                # Calculate refined bounds with buffer
                min_val = max(np.min(nroy_values), self.parameter_bounds[param_name][0])
                max_val = min(np.max(nroy_values), self.parameter_bounds[param_name][1])

                if min_val >= max_val:
                    raise ValueError("Degenerate bounds")

                buffer = (max_val - min_val) * 0.05
                refined_bounds[param_name] = [
                    max(min_val - buffer, self.parameter_bounds[param_name][0]),
                    min(max_val + buffer, self.parameter_bounds[param_name][1]),
                ]

                print(
                    f"Parameter {param_name}: {self.parameter_bounds[param_name]} -> {refined_bounds[param_name]}"
                )

            except Exception as e:
                print(
                    f"Error refining bounds for {param_name}: {e}. Using original bounds."
                )
                refined_bounds[param_name] = self.parameter_bounds[param_name]

        return refined_bounds

    def _get_initial_values(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get initial MCMC values, preferring NROY samples if available."""
        if not self._has_valid_nroy_samples():
            return None

        try:
            # Sample from NROY points
            sample_idx = np.random.choice(len(self.nroy_samples))
            selected_sample = self.nroy_samples[sample_idx]

            init_values = {}
            for param_name in self.important_params:
                param_idx = self.param_names.index(param_name)
                # Convert to tensor and ensure it's on the correct device
                init_values[param_name] = torch.tensor(
                    float(selected_sample[param_idx]),
                    device=self.device,
                    dtype=torch.float32,
                )

            return init_values

        except Exception as e:
            print(
                f"Error getting NROY initial values: {e}. Using default initialization."
            )
            return None

    def _create_full_params(self, reduced_params: torch.Tensor) -> torch.Tensor:
        """Create full parameter vector with defaults for non-calibrated params."""
        full_params = torch.zeros(len(self.parameter_bounds), device=self.device)

        # Set calibrated parameters
        for i, param_name in enumerate(self.important_params):
            param_idx = self.param_names.index(param_name)
            full_params[param_idx] = reduced_params[i]

        # Set non-calibrated parameters to midpoint
        for i, param_name in enumerate(self.param_names):
            if param_name not in self.important_params:
                bounds = self.parameter_bounds[param_name]
                full_params[i] = (bounds[0] + bounds[1]) / 2

        return full_params

    def model(self):
        """Pyro model for MCMC calibration."""
        reduced_params = []

        for param_name in self.important_params:
            bounds = self.calibration_bounds[param_name]
            param_value = pyro.sample(
                param_name,
                dist.Uniform(
                    torch.tensor(bounds[0], device=self.device),
                    torch.tensor(bounds[1], device=self.device),
                ),
            )
            reduced_params.append(param_value)

        # Create full parameter vector and get predictions
        reduced_params_tensor = torch.stack(reduced_params)
        full_params = self._create_full_params(reduced_params_tensor)

        with torch.no_grad():
            pred_mean = torch.tensor(
                self.emulator.predict(
                    full_params.cpu().numpy().reshape(1, -1)
                ).flatten(),
                device=self.device,
            )

        # setup likelihood from observations

        # Pre-convert observations to tensors
        obs_means = torch.tensor(
            [obs[0] for obs in self.observations.values()], device=self.device
        )
        obs_stds = torch.tensor(
            [obs[1] for obs in self.observations.values()], device=self.device
        )

        # One-liner comparison
        [
            pyro.sample(
                f"obs_{name}", dist.Normal(pred_mean[i], obs_stds[i]), obs=obs_means[i]
            )
            for i, name in enumerate(self.observations.keys())
        ]

    def run_mcmc(
        self,
        num_samples: int = 1000,
        warmup_steps: int = 500,
        num_chains: int = 1,
        use_nroy_init: bool = True,
    ) -> Dict:
        """Run MCMC sampling with optional NROY initialization."""
        nuts_kernel = NUTS(self.model)

        # Get initial parameters if requested and available
        initial_params = None
        if use_nroy_init:
            initial_params = self._get_initial_values()
            if initial_params is None:
                print("Using default MCMC initialization.")
            else:
                print("Initializing MCMC chains from NROY samples...")

        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains,
            initial_params=initial_params,
        )

        print("Running MCMC...")
        mcmc.run()

        # Store and summarize results
        self.mcmc_results = {
            p: samples.cpu().numpy() for p, samples in mcmc.get_samples().items()
        }
        self.mcmc_summary = self._create_summary()

        print("MCMC completed!\nPosterior Summary:")
        print(self.mcmc_summary)
        return self.mcmc_results

    def _create_summary(self) -> pd.DataFrame:
        """Create summary statistics for MCMC results."""
        summary_data = []
        for param, samples in self.mcmc_results.items():
            summary_data.append(
                {
                    "parameter": param,
                    "mean": np.mean(samples),
                    "std": np.std(samples),
                    **{
                        f"q{p}": np.percentile(samples, p)
                        for p in [2.5, 25, 50, 75, 97.5]
                    },
                }
            )
        return pd.DataFrame(summary_data).round(4)

    def get_calibrated_parameters(self) -> Dict[str, float]:
        """Get posterior means of calibrated parameters."""
        return {param: np.mean(samples) for param, samples in self.mcmc_results.items()}

    def predict_with_uncertainty(
        self, X_test: np.ndarray, n_posterior_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using posterior uncertainty."""
        n_samples = len(next(iter(self.mcmc_results.values())))
        sample_indices = np.random.choice(
            n_samples, min(n_posterior_samples, n_samples), False
        )

        predictions = []
        for sample_idx in sample_indices:
            # Get calibrated parameters for this sample
            calibrated_params = {
                p: samples[sample_idx] for p, samples in self.mcmc_results.items()
            }

            # Make predictions for all test points
            sample_predictions = []
            for x_test in X_test:
                param_vector = self._create_param_vector(x_test, calibrated_params)
                pred = self.emulator.predict(param_vector.reshape(1, -1))
                sample_predictions.append(pred)

            predictions.append(np.vstack(sample_predictions))

        predictions = np.array(predictions)
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

    def _create_param_vector(
        self, x_test: np.ndarray, calibrated: Dict[str, float]
    ) -> np.ndarray:
        """Create parameter vector mixing test values and calibrated parameters."""
        params = np.zeros(len(self.parameter_bounds))

        for i, param_name in enumerate(self.param_names):
            if param_name in calibrated:
                params[i] = calibrated[param_name]
            elif i < len(x_test):
                params[i] = x_test[i]
            else:
                # Default to midpoint for missing parameters
                bounds = self.parameter_bounds[param_name]
                params[i] = (bounds[0] + bounds[1]) / 2

        return params

    def compare_with_nroy(self) -> Optional[pd.DataFrame]:
        """Compare MCMC results with NROY bounds."""
        if not self._has_valid_nroy_samples():
            print("No valid NROY samples available for comparison")
            return None

        comparison_data = []
        for param_name in self.important_params:
            param_idx = self.param_names.index(param_name)
            nroy_values = self.nroy_samples[:, param_idx]
            mcmc_values = self.mcmc_results[param_name]

            comparison_data.append(
                {
                    "parameter": param_name,
                    "nroy_min": np.min(nroy_values),
                    "nroy_max": np.max(nroy_values),
                    "nroy_mean": np.mean(nroy_values),
                    "mcmc_mean": np.mean(mcmc_values),
                    "mcmc_std": np.std(mcmc_values),
                    "mcmc_q2.5": np.percentile(mcmc_values, 2.5),
                    "mcmc_q97.5": np.percentile(mcmc_values, 97.5),
                }
            )

        return pd.DataFrame(comparison_data)
