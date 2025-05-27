import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, HMC, Predictive
import os
import matplotlib.pyplot as plt
import corner
from datetime import datetime
import arviz as az
from tqdm.auto import tqdm

class OptimizedPyroMCMCInference:
    """
    Optimized MCMC module for parameter inference with significant performance improvements.
    """
    def __init__(self, emulator, param_names, output_names, observations, bounds=None):
        """Initialize with optimizations for tensor operations and GPU usage."""
        self.emulator = emulator
        self.param_names = param_names
        self.output_names = output_names
        self.observations = observations
        self.bounds = bounds
        
        # Set up device and ensure consistent dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        
        # cache observation tensors
        self.obs_means = torch.tensor(
            [observations[name][0] for name in output_names], 
            dtype=self.dtype, device=self.device
        )
        self.obs_vars = torch.tensor(
            [observations[name][1] for name in output_names],
            dtype=self.dtype, device=self.device
        )
        
        # cache parameter bounds if provided
        if bounds:
            self.bounds_low = torch.tensor(
                [bounds[name][0] for name in param_names],
                dtype=self.dtype, device=self.device
            )
            self.bounds_high = torch.tensor(
                [bounds[name][1] for name in param_names],
                dtype=self.dtype, device=self.device
            )
        else:
            self.bounds_low = self.bounds_high = None
            
        # Cache dimensions
        self.n_params = len(param_names)
        self.n_outputs = len(output_names)
        
        # Try to create GPU-compatible emulator wrapper
        self._test_emulator_interface()
        


    def model(self, observations=None):
        """Optimized probabilistic model with reduced tensor operations."""
        theta = pyro.sample("theta", dist.Normal(0, 1))
        
        predictions = self.emulator.predict(theta)
        

        pyro.sample("obs", dist.Normal(predictions, 0.1), 
                   obs=self.observations)
    def run_mcmc(self, method="nuts", n_samples=1000, n_warmup=500, 
                 initial_params=None, output_dir=None, progress=True, 
                 num_chains=1, n_cores=None, step_size=None, target_accept_prob=0.8):
        """Optimized MCMC with better kernel configurations."""
        # Set up MCMC sampler
        mcmc = MCMC(NUTS(self.model), num_samples=1000)
        
        # Sample from posterior: P(theta | observations)
        mcmc.run()
        
        # Return parameter samples
        return mcmc.get_samples()["theta"]
    
    def predict_with_uncertainty(self, results, n_samples=100, batch_size=32):
        """Optimized uncertainty propagation with batching."""

        
    def get_effective_sample_size(self, results):
        """Calculate effective sample size for convergence assessment."""

                
    def plot_chains(self, results, output_file=None, figsize=(12, 10)):
        """Optimized chain plotting."""
