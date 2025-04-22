import json
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from ModularCirc.Models.NaghaviModel import NaghaviModel
from ModularCirc.Models.NaghaviModel import NaghaviModelParameters
from ModularCirc.Solver import Solver
from tqdm.notebook import tqdm  # For Jupyter notebook progress bar

from autoemulate.simulations import circ_utils
from autoemulate.simulations.base import Simulator


def extract_parameter_ranges(json_file_path):
    """
    Extract parameter ranges from a JSON file and return them in the format:
    {
        "param1": (min_val, max_val),  # MUST be tuple of exactly two floats
        "param2": (min_val, max_val),
        ...
    }
    """
    with open(json_file_path) as file:
        dict_parameters = json.load(file)
        circ_utils.condense_dict_parameters(dict_parameters)
        circ_utils.dict_parameters_condensed_range

    return circ_utils.dict_parameters_condensed_range


class NaghaviSimulator(Simulator):
    def __init__(
        self,
        parameters_range: Dict[str, Tuple[float, float]],
        n_cycles: int = 40,
        dt: float = 0.001,
        output_variables: List[str] = None,
    ):
        """Initialize simulator with parameter ranges"""
        self.n_cycles = n_cycles
        self.dt = dt
        self.time_setup = {
            "name": "HistoryMatching",
            "ncycles": n_cycles,
            "tcycle": 1.0,
            "dt": dt,
            "export_min": 1,
        }

        self._param_bounds = parameters_range
        self._param_names = list(self._param_bounds.keys())

        # If specific outputs are provided, use those; otherwise use an empty list
        if output_variables is not None:
            self._output_variables = output_variables  # Store the original variables
        else:
            self._output_variables = []

        self._output_names = (
            []
        )  # Will be populated with expanded stat names after first simulation
        self._has_sample_forward = False  # Flag to track if a simulation has been run

    @property
    def param_names(self) -> List[str]:
        """List of parameter names"""
        return self._param_names

    @property
    def output_names(self) -> List[str]:
        """List of output names with their statistics (min, max, mean, range)"""
        return self._output_names

    @property
    def output_variables(self) -> List[str]:
        """List of original output variables without statistic suffixes"""
        return self._output_variables

    def sample_inputs(self, n_samples: int) -> List[Dict[str, float]]:
        """Generate random samples within the parameter bounds using Latin Hypercube Sampling"""
        samples = []
        param_count = len(self._param_names)

        # Generate Latin Hypercube samples
        lhs_points = np.zeros((n_samples, param_count))
        for i in range(param_count):
            lhs_points[:, i] = np.random.permutation(np.linspace(0, 1, n_samples))

        # Scale to parameter bounds
        for i in range(n_samples):
            sample = {}
            for j, name in enumerate(self._param_names):
                min_val, max_val = self._param_bounds[name]
                sample[name] = min_val + lhs_points[i, j] * (max_val - min_val)
            samples.append(sample)

        return samples

    def _calculate_output_stats(
        self, output_values: np.ndarray, base_name: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate min, max, average, and range for an output time series and return the stats and their names

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

    def sample_forward(self, params: Dict[str, float]) -> Optional[np.ndarray]:
        """Run simulation and return output statistics as a numpy array"""
        # Set parameters
        parobj = NaghaviModelParameters()
        for param_name, value in params.items():
            if param_name == "T":
                continue
            try:
                obj, param = param_name.split(".")
                parobj._set_comp(obj, [obj], **{param: value})
            except Exception as e:
                print(f"Error setting parameter {param_name}: {e}")
                return None

        # Set cycle time
        t_cycle = params.get("T", 1.0)
        self.time_setup["tcycle"] = t_cycle

        # Run simulation
        try:
            model = NaghaviModel(
                time_setup_dict=self.time_setup, parobj=parobj, suppress_printing=True
            )
            solver = Solver(model=model)
            solver.setup(
                suppress_output=True, optimize_secondary_sv=False, method="LSODA"
            )
            solver.solve()

            if not solver.converged:
                print("Solver did not converge.")
                return None
        except Exception as e:
            print(f"Simulation error: {e}")
            return None

        # Collect and process outputs
        output_stats = []
        output_names = []

        for component_name, component_obj in model.components.items():
            for attr_name in dir(component_obj):
                if (
                    not attr_name.startswith("_")
                    and not attr_name == "kwargs"
                    and not callable(getattr(component_obj, attr_name))
                ):
                    try:
                        attr = getattr(component_obj, attr_name)
                        if hasattr(attr, "values"):
                            full_name = f"{component_name}.{attr_name}"

                            # Check if we should track this variable
                            if (
                                not self._output_variables
                                or full_name in self._output_variables
                            ):
                                values = np.array(attr.values)

                                # Get both stats and stat names
                                stats, stat_names = self._calculate_output_stats(
                                    values, full_name
                                )
                                output_stats.extend(stats)
                                output_names.extend(stat_names)
                    except Exception:
                        continue

        # Always update output names after the first simulation
        if not self._has_sample_forward:
            self._output_names = output_names
            self._has_sample_forward = True

        return np.array(output_stats)

    def run_batch_simulations(
        self, samples: Union[List[Dict[str, float]], pd.DataFrame]
    ) -> np.ndarray:
        """Run batch simulations and return results as a 2D numpy array"""
        if isinstance(samples, pd.DataFrame):
            samples = samples.to_dict(orient="records")

        results = []
        successful = 0

        for sample in tqdm(samples, desc="Running simulations", unit="sample"):
            # Run simulation for each sample
            result = self.sample_forward(sample)
            if result is not None:
                results.append(result)
                successful += 1

        print(
            f"Successfully completed {successful}/{len(samples)} simulations ({successful/len(samples)*100:.1f}%)"
        )

        # Convert results to DataFrame with proper column names
        results_array = np.array(results)
        if len(results_array) > 0:
            return results_array
        else:
            return np.array([])
