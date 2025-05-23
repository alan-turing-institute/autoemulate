from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from ModularCirc.Models.NaghaviModel import NaghaviModel, NaghaviModelParameters
from ModularCirc.Solver import Solver
from torch import Tensor
from tqdm import tqdm

from autoemulate.experimental.data.validation import ValidationMixin
from autoemulate.experimental.types import TensorLike
from autoemulate.experimental_design import LatinHypercube


class Simulator(ABC, ValidationMixin):
    """
    Base class for simulations. All simulators should inherit from this class.
    This class provides the interface and common functionality for different
    simulation implementations.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]],
        output_variables: list[str],
    ):
        """
        Initialize the base simulator with parameter ranges
          and optional output variables.

        Parameters
        ----------
        parameters_range : dict
            Dictionary mapping parameter names to their (min, max) ranges.
        output_variables : list
            Optional list of specific output variables to track.
        """
        # define param names and values
        self._parameters_range = parameters_range
        self._param_names = list(self._parameters_range.keys())
        self._param_bounds = list(self._parameters_range.values())

        # Output configuration
        self._output_variables = (
            output_variables if output_variables is not None else []
        )
        self._output_names: list[str] = []  # Will be populated after first simulation
        self._has_sample_forward = False

    def sample_inputs(self, n: int) -> TensorLike:
        """
        Generate random samples within the parameter bounds using
          Latin Hypercube Sampling.

        Args:
            n: Number of samples to generate

        Returns:
            TensorLike: Random input tensor based from param values
        """
        # Use LatinHypercube from autoemulate.experimental_design
        lhd = LatinHypercube(self._param_bounds)
        sample_array = lhd.sample(n)
        return Tensor(sample_array)

    @abstractmethod
    def _forward(self, x: TensorLike) -> TensorLike:
        """
        Abstract method to perform the forward simulation.

        Parameters
        ----------
        x : TensorLike
            Input parameters into the simulation forward run.

        Returns
        -------
        TensorLike
            Simulated output tensor. Shape = (n_batch, *target_shape).
            For example, if the simulator outputs two simulated variables,
            then the shape would be (1, 2).
        """

    def forward(self, x: TensorLike) -> TensorLike:
        """
        Generate samples from input data using the simulator. Combines the
        abstract method `_forward` with some validation checks.

        Parameters
        ----------
        x : TensorLike | dict
            Input tensor of shape (n_samples, n_features) or a dictionary of parameters.

        Returns
        -------
        TensorLike
            Simulated output tensor.
        """
        y = self.check_matrix(self._forward(self.check_matrix(x)))
        x, y = self.check_pair(x, y)
        return y

    def get_parameter_idx(self, name: str) -> int:
        """
        Get the index of a specific parameter.

        Parameters
        ----------
        name : str
            Name of the parameter to retrieve.

        Returns
        -------
        float
            Index of the specified parameter.
        """
        if name not in self._param_names:
            raise ValueError(f"Parameter {name} not found.")
        return self._param_names.index(name)

    @property
    def param_names(self) -> list[str]:
        """List of parameter names"""
        return self._param_names

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """List of parameter bounds"""
        return self._param_bounds

    @property
    def output_names(self) -> list[str]:
        """List of output names with their statistics"""
        return self._output_names

    @property
    def output_variables(self) -> list[str]:
        """List of original output variables without statistic suffixes"""
        return self._output_variables


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
                columns=np.array(self._output_names),
            )
            # Combine parameters and results
            return pd.concat([df_params, df_results], axis=1)
        if len(results) > 0:
            # If output names are not set or don't match, use generic column names
            result_cols = [f"output_{i}" for i in range(results.shape[1])]
            df_results = pd.DataFrame(results, columns=np.array(result_cols))
            return pd.concat([df_params, df_results], axis=1)
        return df_params

    def run_batch_simulations(self, samples: TensorLike) -> TensorLike:
        """
        Run multiple simulations with different parameters.

        Args:
            samples: List of parameter dictionaries or DataFrame of parameters

        Returns:
            2D array of simulation results
        """
        results = []
        successful = 0

        # Process each sample with progress tracking
        for i in tqdm(range(len(samples)), desc="Running simulations"):
            result = self.forward(samples[i : i + 1])
            if result is not None:
                results.append(result)
                successful += 1

        # Report results
        print(
            f"Successfully completed {successful}/{len(samples)}"
            f" simulations ({successful / len(samples) * 100:.1f}%)"
        )

        # stack results into a 2D array on first dim using torch
        return torch.cat(results, dim=0)
