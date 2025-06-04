import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


class MCMCCalibrator:
    def __init__(
        self,
        emulator,
        sensitivity_results: pd.DataFrame,
        observations: Dict[str, Tuple[float, float]],
        parameter_bounds: Dict[str, List[float]],
        top_n_params: int = 5,
        device: str = "cpu"
    ):
        """        
        Parameters:
        -----------
        emulator : trained emulator model
            The fitted emulator (e.g., your gp_final)
        sensitivity_results : pd.DataFrame
            Results from sensitivity analysis (your 'si' variable)
        observations : Dict[str, Tuple[float, float]]
            Observed data as {output_name: (mean, std)} 
        parameter_bounds : Dict[str, List[float]]
            Parameter bounds as {param_name: [min, max]}
        top_n_params : int
            Number of most sensitive parameters to calibrate
        device : str
            PyTorch device ('cpu' or 'cuda')
        """
        self.emulator = emulator
        self.observations = observations
        self.parameter_bounds = parameter_bounds
        self.device = device
        
        # Get most important parameters from sensitivity analysis
        self.important_params = self._get_important_parameters(
            sensitivity_results, top_n_params
        )
        
        # Create reduced parameter bounds
        self.reduced_bounds = {
            param: parameter_bounds[param] 
            for param in self.important_params
        }
        
        print(f"Calibrating {len(self.important_params)} most important parameters:")
        for param in self.important_params:
            print(f"  - {param}")
    
    def _get_important_parameters(
        self, 
        sensitivity_results: pd.DataFrame, 
        top_n: int
    ) -> List[str]:
        """Get the most important parameters based on total Sobol indices."""
        # Filter for total order indices (ST)
        st_results = sensitivity_results[
            sensitivity_results['index'] == 'ST'
        ].copy()
        
        # Average across outputs for each parameter
        param_importance = st_results.groupby('parameter')['value'].mean()
        
        # Get top N parameters
        top_params = param_importance.nlargest(top_n).index.tolist()
        
        return top_params
    
    def _create_full_params(self, reduced_params: torch.Tensor) -> torch.Tensor:
        """
        Create full parameter vector by filling non-calibrated params with defaults.
        """
        # Get all parameter names in original order
        all_param_names = list(self.parameter_bounds.keys())
        full_params = torch.zeros(len(all_param_names), device=self.device)
        
        # Fill with mid-range values for non-calibrated parameters
        for i, param_name in enumerate(all_param_names):
            if param_name in self.important_params:
                # Get index in reduced parameter vector
                reduced_idx = self.important_params.index(param_name)
                full_params[i] = reduced_params[reduced_idx]
            else:
                # Use midpoint of range for non-calibrated parameters
                bounds = self.parameter_bounds[param_name]
                full_params[i] = (bounds[0] + bounds[1]) / 2.0
        
        return full_params
    
    def model(self, obs_data: torch.Tensor):
        """Pyro model for MCMC sampling."""
        # Sample reduced parameters with uniform priors
        reduced_params = []
        for param_name in self.important_params:
            bounds = self.reduced_bounds[param_name]
            param_val = pyro.sample(
                param_name,
                dist.Uniform(
                    torch.tensor(bounds[0], device=self.device),
                    torch.tensor(bounds[1], device=self.device)
                )
            )
            reduced_params.append(param_val)
        
        reduced_params_tensor = torch.stack(reduced_params)
        
        # Create full parameter vector
        full_params = self._create_full_params(reduced_params_tensor)
        
        # Get emulator prediction
        with torch.no_grad():
            # Convert to numpy for emulator prediction
            params_np = full_params.cpu().numpy().reshape(1, -1)
            pred_mean = self.emulator.predict(params_np).flatten()
            pred_mean_tensor = torch.tensor(pred_mean, device=self.device)
        
        # Likelihood - compare with observations
        for i, (output_name, (obs_mean, obs_std)) in enumerate(self.observations.items()):
            pyro.sample(
                f"obs_{output_name}",
                dist.Normal(pred_mean_tensor[i], torch.tensor(obs_std, device=self.device)),
                obs=torch.tensor(obs_mean, device=self.device)
            )
    
    def run_mcmc(
        self,
        num_samples: int = 1000,
        warmup_steps: int = 500,
        num_chains: int = 1
    ) -> Dict:
        """
        Run MCMC sampling.
        
        Parameters:
        -----------
        num_samples : int
            Number of MCMC samples
        warmup_steps : int  
            Number of warmup steps
        step_size : float
            HMC step size
        num_chains : int
            Number of chains
            
        Returns:
        --------
        Dict with posterior samples
        """
        # Prepare observed data tensor
        obs_means = [obs[0] for obs in self.observations.values()]
        obs_data = torch.tensor(obs_means, device=self.device)
        
        # Set up HMC
        nuts_kernel = NUTS(self.model)

        
        # Run MCMC
        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains
        )
        
        print("Running MCMC...")
        mcmc.run(obs_data)
        
        # Get samples
        samples = mcmc.get_samples()
        
        # Convert to numpy and create results dictionary
        results = {}
        for param_name in self.important_params:
            results[param_name] = samples[param_name].cpu().numpy()
        
        self.mcmc_results = results
        self.mcmc_summary = self._summarize_results(results)
        
        print("MCMC completed!")
        print("\nPosterior Summary:")
        print(self.mcmc_summary)
        
        return results
    
    def _summarize_results(self, results: Dict) -> pd.DataFrame:
        """Create summary statistics for MCMC results."""
        summary_data = []
        
        for param_name, samples in results.items():
            summary_data.append({
                'parameter': param_name,
                'mean': np.mean(samples),
                'std': np.std(samples),
                'q2.5': np.percentile(samples, 2.5),
                'q25': np.percentile(samples, 25),
                'q50': np.percentile(samples, 50),
                'q75': np.percentile(samples, 75),
                'q97.5': np.percentile(samples, 97.5)
            })
        
        return pd.DataFrame(summary_data).round(4)
    
    def get_calibrated_parameters(self) -> Dict[str, float]:
        """Get point estimates (posterior means) of calibrated parameters."""

        return {
            param: np.mean(samples) 
            for param, samples in self.mcmc_results.items()
        }
    
    def predict_with_uncertainty(
        self, 
        X_test: np.ndarray, 
        n_posterior_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using posterior uncertainty.
        """
        
        # Sample from posterior
        sample_indices = np.random.choice(
            len(list(self.mcmc_results.values())[0]), 
            size=n_posterior_samples, 
            replace=False
        )
        
        predictions = []
        
        for idx in sample_indices:
            # Get posterior sample for calibrated parameters
            calibrated_values = {
                param: samples[idx] 
                for param, samples in self.mcmc_results.items()
            }
            
            # Create full parameter vector for each test point
            test_predictions = []
            for x_test in X_test:
                # Fill in calibrated parameters
                all_param_names = list(self.parameter_bounds.keys())
                full_params = np.zeros(len(all_param_names))
                
                for i, param_name in enumerate(all_param_names):
                    if param_name in calibrated_values:
                        full_params[i] = calibrated_values[param_name]
                    else:
                        # Use provided test value or midpoint
                        if i < len(x_test):
                            full_params[i] = x_test[i]
                        else:
                            bounds = self.parameter_bounds[param_name]
                            full_params[i] = (bounds[0] + bounds[1]) / 2.0
                
                pred = self.emulator.predict(full_params.reshape(1, -1))
                test_predictions.append(pred.flatten())
            
            predictions.append(np.array(test_predictions))
        
        predictions = np.array(predictions)
        
        # Calculate mean and std across posterior samples
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred

