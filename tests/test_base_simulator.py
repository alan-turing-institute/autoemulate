import pytest
import numpy as np
from typing import Dict, List, Optional
from autoemulate.simulations.base_simulator import BaseSimulator

# Mock implementation of BaseSimulator for testing
class MockSimulator(BaseSimulator):
    """Mock implementation of BaseSimulator for testing purposes"""
    
    def __init__(self, param_ranges):
        self._param_bounds = param_ranges
        self._param_names_list = list(param_ranges.keys())
        self._output_names_list = ["output1", "output2"]
    
    @property
    def param_names(self) -> List[str]:
        return self._param_names_list
    
    @property
    def output_names(self) -> List[str]:
        return self._output_names_list
    
    def generate_initial_samples(self, n_samples: int) -> List[Dict[str, float]]:
        """Generate mock samples"""
        samples = []
        for i in range(n_samples):
            sample = {}
            for name in self._param_names_list:
                min_val, max_val = self._param_bounds[name]
                sample[name] = min_val + (max_val - min_val) * np.random.random()
            samples.append(sample)
        return samples
    
    def run_simulation(self, params: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Run mock simulation"""
        # Simple deterministic output based on input parameters
        try:
            param_sum = sum(params.values())
            return {
                "output1": param_sum / len(params),
                "output2": param_sum * 2
            }
        except Exception:
            return None

