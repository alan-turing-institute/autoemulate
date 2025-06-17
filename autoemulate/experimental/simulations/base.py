from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from autoemulate.experimental.data.utils import ValidationMixin
from autoemulate.experimental.types import TensorLike
from autoemulate.experimental_design import LatinHypercube


class Simulator(ABC, ValidationMixin):
    """
    Base class for simulations. All simulators should inherit from this class.
    This class provides the interface and common functionality for different
    simulation implementations.
    """

    def __init__(
        self, parameters_range: dict[str, tuple[float, float]], output_names: list[str]
    ):
        """
        Parameters
        ----------
        parameters_range : dict[str, tuple[float, float]]
            Dictionary mapping input parameter names to their (min, max) ranges.
        output_names: list[str]
            List of output parameters' names.
        """
        self._parameters_range = parameters_range
        self._param_names = list(parameters_range.keys())
        self._param_bounds = list(parameters_range.values())
        self._output_names = output_names
        self._in_dim = len(self.param_names)
        self._out_dim = len(self.output_names)
        self._has_sample_forward = False

    @property
    def parameters_range(self) -> dict[str, tuple[float, float]]:
        """Dictionary mapping input parameter names to their (min, max) ranges."""
        return self._parameters_range

    @property
    def param_names(self) -> list[str]:
        """List of parameter names."""
        return self._param_names

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """List of parameter bounds."""
        return self._param_bounds

    @property
    def output_names(self) -> list[str]:
        """List of output parameter names."""
        return self._output_names

    @property
    def in_dim(self) -> int:
        """Input dimensionality."""
        return self._in_dim

    @property
    def out_dim(self) -> int:
        """Output dimensionality."""
        return self._out_dim

    def sample_inputs(self, n_samples: int) -> TensorLike:
        """
        Generate random samples using Latin Hypercube Sampling.

        Parameters
        ----------
            n_samples: int
                Number of samples to generate.

        Returns
        -------
        TensorLike
            Parameter samples (column order is given by self.param_names)
        """

        lhd = LatinHypercube(self.param_bounds)
        sample_array = lhd.sample(n_samples)
        # TODO: have option to set dtype and ensure consistency throughout codebase?
        # added here as method was returning float64 and elsewhere had tensors of
        # float32 and this caused issues
        return torch.tensor(sample_array, dtype=torch.float32)

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
            Simulated output tensor. Shape = (1, self.out_dim).
            For example, if the simulator outputs two simulated variables,
            then the shape would be (1, 2).
        """

    def forward(self, x: TensorLike) -> TensorLike:
        """
        Generate samples from input data using the simulator. Combines the
        abstract method `_forward` with some validation checks.

        Parameters
        ----------
        x : TensorLike
            Input tensor of shape (n_samples, self.in_dim).

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

        Parameters
        ----------
        samples: TensorLike
            Tensor of input parameters to make predictions for.

        Returns:
        -------
        TensorLike
            Tensor of simulation results of shape (n_batch, self.out_dim).
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
