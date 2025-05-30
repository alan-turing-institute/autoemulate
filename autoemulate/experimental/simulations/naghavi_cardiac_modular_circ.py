import numpy as np
import pandas as pd
from ModularCirc.Models.NaghaviModel import NaghaviModel, NaghaviModelParameters
from ModularCirc.Solver import Solver
from torch import Tensor

from autoemulate.experimental.simulations.base import Simulator
from autoemulate.experimental.types import TensorLike


class NaghaviSimulator(Simulator):
    def __init__(
        self,
        parameters_range,
        output_variables,
        n_cycles: int = 40,
        dt: float = 0.001,
    ):
        """
        Initialize the Naghavi simulator with specific parameters.
        Some default parameter ranges can be found
        autoemulate.simulations.naghavi_cardiac_ModularCirc.py

        Parameters
        ----------
        parameters_range : dict
            Dictionary mapping parameter names to (min, max) tuples.
        output_variables : list
            List of specific output variables to track.
        n_cycles : int
            Number of simulation cycles.
        dt : float
            Time step size.
        """
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

    def _forward(self, x: TensorLike) -> TensorLike:
        """
        Run a single Naghavi model simulation and return output statistics.

        Args:
            x: TensorLike
                Input parameters for the simulation

        Returns:
            Array of output statistics or None if simulation fails
        """
        if x.shape[-1] != len(self._param_names):
            raise ValueError(
                f"Input x must have the same shape as the number of parameters:"
                f" {len(self._param_names)}"
            )

        # Drop first dim
        x = x.squeeze(0)

        # Set parameter object
        parobj = NaghaviModelParameters()

        for i, param_name in enumerate(self._param_names):
            if param_name == "T":
                continue

            obj, param = param_name.split(".")
            value = x[i].numpy()
            parobj._set_comp(obj, [obj], **{param: value})

        # Set cycle time
        t_cycle = (
            x[self.get_parameter_idx("T")].item() if "T" in self._param_names else 1.0
        )

        self.time_setup["tcycle"] = t_cycle

        # Run simulation
        model = NaghaviModel(
            time_setup_dict=self.time_setup, parobj=parobj, suppress_printing=True
        )
        solver = Solver(model=model)
        solver.setup(suppress_output=True, optimize_secondary_sv=False, method="LSODA")
        solver.solve()

        if not solver.converged:
            err_msg = "Solver did not converge"
            raise Exception(err_msg)

        # Collect and process outputs
        output_stats = []
        output_names = []

        for component_name, component_obj in model.components.items():
            for attr_name in dir(component_obj):
                if (
                    not attr_name.startswith("_")
                    and attr_name != "kwargs"
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

        # Convert output stats to a tensor
        return Tensor(output_stats).unsqueeze(0)

    def _calculate_output_stats(
        self, output_values: np.ndarray, base_name: str
    ) -> tuple[np.ndarray, list[str]]:
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
        self, samples: list[dict[str, float]], results: np.ndarray
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
            df_results = pd.DataFrame(
                results,
                columns=self._output_names,  # type: ignore[reportArgumentType] Pyright doesn't seem to like having a list of strings as type of columns.
            )
            # Combine parameters and results
            return pd.concat([df_params, df_results], axis=1)
        if len(results) > 0:
            # If output names are not set or don't match, use generic column names
            result_cols = [f"output_{i}" for i in range(results.shape[1])]
            df_results = pd.DataFrame(results, columns=result_cols)  # type: ignore[reportArgumentType] Pyright doesn't seem to like having a list of strings as type of columns.
            return pd.concat([df_params, df_results], axis=1)
        return df_params
