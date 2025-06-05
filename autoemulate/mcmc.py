from typing import Dict
from typing import List
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
        top_n_params: int = 5,
        device: str = "cpu",
    ):
        self.emulator = emulator
        self.observations = observations
        self.device = device
        self.parameter_bounds = parameter_bounds
        # Get important parameters and their bounds
        self.important_params = self._get_important_parameters(
            sensitivity_results, top_n_params
        )
        self.reduced_bounds = {p: parameter_bounds[p] for p in self.important_params}

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

    def _create_full_params(self, reduced_params: torch.Tensor) -> torch.Tensor:
        """Create full parameter vector with defaults for non-calibrated params."""
        full_params = torch.zeros(len(self.parameter_bounds), device=self.device)
        for i, param_name in enumerate(self.parameter_bounds):
            if param_name in self.important_params:
                full_params[i] = reduced_params[self.important_params.index(param_name)]
            else:
                bounds = self.parameter_bounds[param_name]
                full_params[i] = (bounds[0] + bounds[1]) / 2
        return full_params

    def model(self):
        """Pyro model for MCMC sampling."""
        reduced_params = [
            pyro.sample(
                param_name,
                dist.Uniform(
                    torch.tensor(bounds[0], device=self.device),
                    torch.tensor(bounds[1], device=self.device),
                ),
            )
            for param_name, bounds in self.reduced_bounds.items()
        ]

        full_params = self._create_full_params(torch.stack(reduced_params))

        with torch.no_grad():
            pred_mean = torch.tensor(
                self.emulator.predict(
                    full_params.cpu().numpy().reshape(1, -1)
                ).flatten(),
                device=self.device,
            )

        for i, (output_name, (obs_mean, obs_std)) in enumerate(
            self.observations.items()
        ):
            pyro.sample(
                f"obs_{output_name}",
                dist.Normal(pred_mean[i], torch.tensor(obs_std, device=self.device)),
                obs=torch.tensor(obs_mean, device=self.device),
            )

    def run_mcmc(
        self, num_samples: int = 1000, warmup_steps: int = 500, num_chains: int = 1
    ) -> Dict:
        """Run MCMC sampling and return results."""
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains,
        )

        print("Running MCMC...")
        mcmc.run()

        self.mcmc_results = {
            p: samples.cpu().numpy() for p, samples in mcmc.get_samples().items()
        }
        self.mcmc_summary = self._summarize_results(self.mcmc_results)

        print("MCMC completed!\nPosterior Summary:")
        print(self.mcmc_summary)
        return self.mcmc_results

    def _summarize_results(self, results: Dict) -> pd.DataFrame:
        """Create summary statistics for MCMC results."""
        return pd.DataFrame(
            [
                {
                    "parameter": param,
                    "mean": np.mean(samples),
                    "std": np.std(samples),
                    **{
                        f"q{p}": np.percentile(samples, p)
                        for p in [2.5, 25, 50, 75, 97.5]
                    },
                }
                for param, samples in results.items()
            ]
        ).round(4)

    def get_calibrated_parameters(self) -> Dict[str, float]:
        """Get posterior means of calibrated parameters."""
        return {param: np.mean(samples) for param, samples in self.mcmc_results.items()}

    def predict_with_uncertainty(
        self, X_test: np.ndarray, n_posterior_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using posterior uncertainty."""
        sample_indices = np.random.choice(
            len(next(iter(self.mcmc_results.values()))), n_posterior_samples, False
        )

        predictions = np.array(
            [
                self._predict_single_sample(X_test, sample_idx)
                for sample_idx in sample_indices
            ]
        )

        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

    def _predict_single_sample(self, X_test: np.ndarray, sample_idx: int) -> np.ndarray:
        """Helper for predict_with_uncertainty."""
        calibrated = {
            p: samples[sample_idx] for p, samples in self.mcmc_results.items()
        }

        return np.vstack(
            [
                self.emulator.predict(
                    self._create_param_vector(x_test, calibrated).reshape(1, -1)
                )
                for x_test in X_test
            ]
        )

    def _create_param_vector(
        self, x_test: np.ndarray, calibrated: Dict[str, float]
    ) -> np.ndarray:
        """Create parameter vector mixing test values and calibrated parameters."""
        params = np.zeros(len(self.parameter_bounds))
        for i, param_name in enumerate(self.parameter_bounds):
            params[i] = calibrated.get(
                param_name,
                x_test[i]
                if i < len(x_test)
                else sum(self.parameter_bounds[param_name]) / 2,
            )
        return params
