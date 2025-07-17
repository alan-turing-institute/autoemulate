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

from autoemulate.experimental.simulations.base import Simulator

### UTILS

dict_parameters_condensed_range = dict()
dict_parameters_condensed_single = dict()


def condense_dict_parameters(dict_param: dict, prev: str = ""):
    for key, val in dict_param.items():
        if len(prev) > 0:
            new_key = prev.split(".")[-1] + "." + key
        else:
            new_key = key
        if isinstance(val, dict):
            condense_dict_parameters(val, new_key)
        else:
            if len(val) > 1:
                value, r = val
                dict_parameters_condensed_range[new_key] = tuple(np.array(r) * value)
            else:
                dict_parameters_condensed_single[new_key] = val[0]
    return


def extract_parameter_ranges(json_file_path: str):
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
        condense_dict_parameters(dict_parameters)

    return dict_parameters_condensed_range


### SIMULATOR


class NaghaviSimulator(Simulator):
    def __init__(
        self,
        parameters_range: Dict[str, Tuple[float, float]],
        output_variables: Optional[List[str]] = None,
        n_cycles: int = 40,
        dt: float = 0.001,
    ):
        """
        Initialize the Naghavi simulator.

        Parameters
        ----------
            parameters_range: Dictionary mapping parameter names to (min, max) tuples
            n_cycles: Number of simulation cycles
            dt: Time step size
            output_variables: Optional list of specific output variables to track
        """
        # Initialize the base class
        output_names = []
        super().__init__(parameters_range, output_names)
        self._output_variables = output_variables

        # Naghavi-specific attributes
        self.n_cycles = n_cycles
        self.dt = dt
        self.time_setup = {
            "name": "HistoryMatching",
            "ncycles": n_cycles,
            "tcycle": 1.0,
            "dt": dt,
            "export_min": 1,
        }

    @property
    def output_variables(self) -> List[str]:
        """List of original output variables without statistic suffixes"""
        return self._output_variables

    def _forward(self, params: Dict[str, float]) -> Optional[np.ndarray]:
        """
        TODO: params -> x: TensorLike

        Run a single Naghavi model simulation and return output statistics.

        Parameters
        ----------
        params: Dictionary of parameter values

        Returns
        -------
            Array of output statistics or None if simulation fails
        """
        # Set parameters
        # TODO: has to work with tensor of values in order of self.param_names
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

                                # Use the base class method to calculate statistics
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

    def _calculate_output_stats(
        self, output_values: np.ndarray, base_name: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate statistics for an output time series.

        Parameters
        ----------
            output_values: Array of time series values
            base_name: Base name of the output variable

        Returns
        -------
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

        Parameters
        ----------
            samples: List of parameter dictionaries
            results: 2D array of simulation results

        Returns
        -------
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
