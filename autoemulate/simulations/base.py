from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from autoemulate.experimental_design import LatinHypercube


class Simulator(ABC):
    """
    Abstract base class for simulation models.

    This class provides the interface and common functionality for different
    simulation implementations. Specific simulators should inherit from this class
    and implement the required abstract methods.
    """

    def __init__(
        self,
        parameters_range: Dict[str, Tuple[float, float]],
        output_variables: Optional[List[str]] = None,
    ):
        """
        Initialize the base simulator with parameter ranges and optional output variables.

        Args:
            parameters_range: Dictionary mapping parameter names to their (min, max) ranges
            output_variables: Optional list of specific output variables to track
        """
        self._param_bounds = parameters_range
        self._param_names = list(self.param_bounds.keys())

        # Output configuration
        self._output_variables = (
            output_variables if output_variables is not None else []
        )
        self._output_names = []  # Will be populated after first simulation
        self._has_sample_forward = False

    @property
    def param_names(self) -> List[str]:
        """List of parameter names"""
        return self._param_names

    @property
    def output_names(self) -> List[str]:
        """List of output names with their statistics"""
        return self._output_names

    @property
    def output_variables(self) -> List[str]:
        """List of original output variables without statistic suffixes"""
        return self._output_variables

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get the parameter bounds"""
        return self._param_bounds

    @abstractmethod
    def sample_forward(self, params: Dict[str, float]) -> Optional[np.ndarray]:
        """
        Run a single simulation with the given parameters and return output statistics.

        Args:
            params: Dictionary of parameter values

        Returns:
            Array of output statistics or None if simulation fails
        """
        pass

    def sample_inputs(self, n_samples: int) -> np.ndarray:
        """
        Generate random samples within the parameter bounds using Latin Hypercube Sampling.

        Args:
            n_samples: Number of samples to generate

        Returns:
            np.ndarray: Parameter samples (column order is given by self.param_names)
        """

        # TODO: check that sample_array has params self.param_names order
        lhd = LatinHypercube([self.param_bounds[name] for name in self.param_names])
        sample_array = lhd.sample(n_samples)
        return sample_array

    def convert_samples(
        self, samples: Union[np.ndarray, pd.DataFrame]
    ) -> list[dict[str, float]]:
        """
        Convert the sample array or dataframe to a list of parameter dictionaries.
        """
        if isinstance(samples, pd.DataFrame):
            return samples.to_dict(orient="records")

        elif isinstance(samples, np.ndarray):
            samples_list = []
            for i in range(len(samples)):
                sample = {}
                for j, name in enumerate(self.param_names):
                    sample[name] = samples[i, j]
                samples_list.append(sample)
            return samples_list
        else:
            raise ValueError("samples must be an array or pandas dataframe")

    def run_batch_simulations(self, samples: np.ndarray) -> np.ndarray:
        """
        Run multiple simulations with different parameters.

        Args:
            samples: List of parameter dictionaries or DataFrame of parameters

        Returns:
            2D array of simulation results
        """

        # convert inputs to list of dictionaries
        samples = self.convert_samples(samples)

        results = []
        successful = 0

        # Import tqdm here to avoid potential import errors
        try:
            from tqdm.auto import tqdm as progress_bar
        except ImportError:
            # Fallback to a simple progress tracking if tqdm is not available
            def progress_bar(iterable, **kwargs):
                return iterable

        # Process each sample with progress tracking
        for sample in progress_bar(samples, desc="Running simulations", unit="sample"):
            result = self.sample_forward(sample)
            if result is not None:
                results.append(result)
                successful += 1

        # Report results
        print(
            f"Successfully completed {successful}/{len(samples)} simulations ({successful/len(samples)*100:.1f}%)"
        )

        # Convert results to numpy array
        if results:
            return np.array(results)
        else:
            return np.array([])

    def _calculate_output_stats(
        self, output_values: np.ndarray, base_name: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate statistics for an output time series.

        Args:
            output_values: Array of time series values
            base_name: Base name of the output variable

        Returns:
            Tuple of (stats_array, stat_names)
        """
        stats = np.array(
            [
                np.min(output_values),
                np.max(output_values),
                np.mean(output_values),
                np.max(output_values) - np.min(output_values),
            ]
        )

        stat_names = [
            f"{base_name}_min",
            f"{base_name}_max",
            f"{base_name}_mean",
            f"{base_name}_range",
        ]

        return stats, stat_names

    def get_results_dataframe(
        self, samples: List[Dict[str, float]], results: np.ndarray
    ) -> pd.DataFrame:
        """
        Create a DataFrame with both input parameters and output results.

        Args:
            samples: List of parameter dictionaries
            results: 2D array of simulation results

        Returns:
            DataFrame with parameters and results
        """
        # Create DataFrame with parameters
        df_params = pd.DataFrame(samples)

        # Create DataFrame with results
        if len(results) > 0 and len(self._output_names) == results.shape[1]:
            df_results = pd.DataFrame(results, columns=self._output_names)
            # Combine parameters and results
            return pd.concat([df_params, df_results], axis=1)
        elif len(results) > 0:
            # If output names are not set or don't match, use generic column names
            result_cols = [f"output_{i}" for i in range(results.shape[1])]
            df_results = pd.DataFrame(results, columns=result_cols)
            return pd.concat([df_params, df_results], axis=1)
        else:
            return df_params
