import json
from itertools import product

import numpy as np
import pandas as pd
import torch
from ModularCirc.Models.NaghaviModel import NaghaviModel, NaghaviModelParameters
from ModularCirc.Solver import Solver

from autoemulate.experimental.simulations.base import Simulator
from autoemulate.experimental.types import NumpyLike, TensorLike

# ==================================
# PARAMETER UTILS
# ==================================


def _condense_dict_parameters(
    dict_param: dict, condensed_range: dict, condensed_single: dict, prev: str = ""
) -> None:
    """
    Recursively condense a nested dictionary into two flat dictionaries:
    - `condensed_range`: Stores keys with value ranges (tuples of two floats).
    - `condensed_single`: Stores keys with single values (floats).
    """
    for key, val in dict_param.items():
        if len(prev) > 0:
            new_key = prev.split(".")[-1] + "." + key
        else:
            new_key = key
        if isinstance(val, dict):
            _condense_dict_parameters(val, condensed_range, condensed_single, new_key)
        else:
            if len(val) > 1:
                value, r = val
                condensed_range[new_key] = tuple(np.array(r) * value)
            else:
                condensed_single[new_key] = val[0]
    return


def extract_parameter_ranges(
    json_file_path: str,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """
    Extract parameter values from a JSON file and return two dictionaries:
     - `condensed_range`: Parameters with value ranges (tuples of two floats).
     - `condensed_single`: Parameters with single values (floats).

    Parameters
    ----------
    json_file_path: str
        Path to the JSON file to extract parameter values from.

    Returns
    -------
    tuple[dict, dict]
        Dictionary of parameter ranges and dictionary of single values.
    """
    with open(json_file_path) as file:
        dict_parameters = json.load(file)

    condensed_single = {}
    condensed_range = {}
    _condense_dict_parameters(dict_parameters, condensed_range, condensed_single)

    return condensed_range, condensed_single


# ==================================
# SIMULATOR
# ==================================


class NaghaviSimulator(Simulator):
    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]],
        output_variables: list[str] | None = None,
        log_level: str = "progress_bar",
        n_cycles: int = 40,
        dt: float = 0.001,
    ):
        """
        Initialize the Naghavi simulator.

        Parameters
        ----------
        parameters_range: dict[str, tuple[float, float]]
            Dictionary mapping input parameter names to their (min, max) ranges.
        output_variables: list[str]
            Optional list of specific output variables to track. Defaults to None.
        log_level: str
            Logging level for the simulator. Can be one of:
            - "progress_bar": shows a progress bar during batch simulations
            - "debug": shows debug messages
            - "info": shows informational messages
            - "warning": shows warning messages
            - "error": shows error messages
            - "critical": shows critical messages
        n_cycles: int
            Number of simulation cycles.
        dt: float
            Time step size.
        """
        # Initialize the base class
        if self.output_variables is not None:
            output_names = self._create_output_names(output_variables)
        else:
            # The Naghavi heart model is structured as components (e.g., lv is the
            # left ventricle) with 4 variables simulated in each component
            components = ["ao", "art", "ven", "av", "mv", "la", "lv"]
            variables = ["P_i", "P_o", "Q_i", "Q_o"]
            output_variables = [
                f"{component}.{variable}"
                for component, variable in product(components, variables)
            ]
            output_names = self._create_output_names(output_variables)
        super().__init__(parameters_range, output_names, log_level)
        assert output_variables is not None
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
    def output_variables(self) -> list[str]:
        """List of original output variables without statistic suffixes."""
        return self._output_variables

    def _forward(self, x: TensorLike) -> TensorLike:
        # TODO: has to work with tensor of values in order of self.param_names

        parobj = NaghaviModelParameters()
        for i, param_name in enumerate(self.param_names):
            obj, param = param_name.split(".")
            value = x[i].item()
            parobj._set_comp(obj, [obj], **{param: value})

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
        for output_var in self.output_variables:
            component, variable = output_var.split(".")
            values = np.array(getattr(model.components[component], variable).values)
            stats = self._calculate_output_stats(values, output_var)
            output_stats.extend(stats)

        return torch.tensor(output_stats)

    def _create_output_names(self, output_vars: list[str]):
        output_names = []
        for base_name in output_vars:
            stat_names = [
                f"{base_name}_min",
                f"{base_name}_max",
                f"{base_name}_mean",
                f"{base_name}_range",
            ]
            output_names.extend(stat_names)
        return output_names

    def _calculate_output_stats(self, output_values: NumpyLike) -> NumpyLike:
        """
        Calculate summary statistics for an output time series.

        Parameters
        ----------
        output_values: NumpyLike
            Array of time series values.

        Returns
        -------
        NumpyLike
            Array of output statistics.
        """
        return np.array(
            [
                np.min(output_values),
                np.max(output_values),
                np.mean(output_values),
                np.max(output_values) - np.min(output_values),
            ]
        )

    def get_results_dataframe(
        self, samples: list[dict[str, float]], results: NumpyLike
    ) -> pd.DataFrame:
        """
        TODO: check this

        Create a DataFrame with both input parameters and output results.

        Parameters
        ----------
        samples: list[dict[str, float]]
            List of parameter dictionaries.
        results: NumpyLike
            2D array of simulation results

        Returns
        -------
        pd.DataFrame
            DataFrame with parameters and results.
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
