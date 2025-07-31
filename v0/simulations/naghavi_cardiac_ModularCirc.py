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

from autoemulate.experimental_design import LatinHypercube
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
        output_variables: Optional[List[str]] = None,
    ):
        """
        Initialize the Naghavi simulator.

        Args:
            parameters_range: Dictionary mapping parameter names to (min, max) tuples
            n_cycles: Number of simulation cycles
            dt: Time step size
            output_variables: Optional list of specific output variables to track
        """
        # Initialize the base class
        super().__init__(parameters_range, output_variables)

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

    def sample_forward(self, params: Dict[str, float]) -> Optional[np.ndarray]:
        """
        Run a single Naghavi model simulation and return output statistics.

        Args:
            params: Dictionary of parameter values

        Returns:
            Array of output statistics or None if simulation fails
        """
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
