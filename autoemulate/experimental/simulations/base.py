from abc import ABC, abstractmethod

import torch
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
        return torch.Tensor(sample_array)

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

    def forward_batch(self, samples: TensorLike) -> TensorLike:
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
