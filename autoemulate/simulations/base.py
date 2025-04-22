from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np


class Simulator(ABC):
    """Abstract base class for simulators used in history matching"""

    @abstractmethod
    def run_simulation(self, params: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Run simulation with given parameters and return outputs

        Args:
            params: Dictionary of parameter name-value pairs

        Returns:
            Dictionary of output name-value pairs or None if simulation fails
        """
        pass

    @abstractmethod
    def generate_initial_samples(self, n_samples: int) -> List[Dict[str, float]]:
        """
        Generate initial parameter samples

        Args:
            n_samples: Number of samples to generate

        Returns:
            List of parameter dictionaries
        """
        pass

    @property
    @abstractmethod
    def param_names(self) -> List[str]:
        """List of parameter names"""
        pass

    @property
    @abstractmethod
    def output_names(self) -> List[str]:
        """List of output names"""
        pass
