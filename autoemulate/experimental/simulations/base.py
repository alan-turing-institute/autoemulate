import logging
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from autoemulate.experimental.data.utils import ValidationMixin, set_random_seed
from autoemulate.experimental.logging_config import get_configured_logger
from autoemulate.experimental.simulations.experimental_design import LatinHypercube
from autoemulate.experimental.types import TensorLike

logger = logging.getLogger("autoemulate")


class Simulator(ABC, ValidationMixin):
    """
    Base class for simulations. All simulators should inherit from this class.
    This class provides the interface and common functionality for different
    simulation implementations.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]],
        output_names: list[str],
        log_level: str = "progress_bar",
    ):
        """
        Parameters
        ----------
        parameters_range: dict[str, tuple[float, float]]
            Dictionary mapping input parameter names to their (min, max) ranges.
        output_names: list[str]
            List of output parameters' names.
        log_level: str
            Logging level for the simulator. Can be one of:
            - "progress_bar": shows a progress bar during batch simulations
            - "debug": shows debug messages
            - "info": shows informational messages
            - "warning": shows warning messages
            - "error": shows error messages
            - "critical": shows critical messages
        """
        self._parameters_range = parameters_range
        self._param_names = list(parameters_range.keys())
        self._param_bounds = list(parameters_range.values())
        self._output_names = output_names
        self._in_dim = len(self.param_names)
        self._out_dim = len(self.output_names)
        self._has_sample_forward = False
        self.logger, self.progress_bar = get_configured_logger(log_level)

    @classmethod
    def simulator_name(cls) -> str:
        return cls.__name__

    @property
    def parameters_range(self) -> dict[str, tuple[float, float]]:
        """Dictionary mapping input parameter names to their (min, max) ranges."""
        return self._parameters_range

    @parameters_range.setter
    def parameters_range(
        self, parameters_range: dict[str, tuple[float, float]]
    ) -> None:
        """Set the range of input parameters for the simulator.

        Parameters
        ----------
        parameters_range: dict[str, tuple[float, float]]
            Dictionary mapping input parameter names to their (min, max) ranges.
        """
        self._parameters_range = parameters_range
        self._param_names = list(parameters_range.keys())
        self._param_bounds = list(parameters_range.values())
        self._in_dim = len(self.param_names)

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

    @output_names.setter
    def output_names(self, output_names: list[str]) -> None:
        """Set the names of output parameters for the simulator.

        This setter allows renaming the output parameters but does not allow
        changing the number of outputs (dimensionality is fixed after initialization).

        Parameters
        ----------
        output_names: list[str]
            List of output parameter names. Must have the same length as the current
            number of outputs.

        Raises
        ------
        ValueError
            If the number of output names differs from the simulator's fixed output
            dimension.
        """
        if len(output_names) != self._out_dim:
            raise ValueError(
                f"Number of output names ({len(output_names)}) must match "
                f"simulator output dimension ({self._out_dim}). Cannot change "
                f"dimensionality after initialization."
            )
        self._output_names = output_names

    @property
    def in_dim(self) -> int:
        """Input dimensionality."""
        return self._in_dim

    @property
    def out_dim(self) -> int:
        """Output dimensionality."""
        return self._out_dim

    def sample_inputs(
        self, n_samples: int, random_seed: int | None = None
    ) -> TensorLike:
        """
        Generate random samples using Latin Hypercube Sampling.

        Parameters
        ----------
        n_samples: int
            Number of samples to generate.
        random_seed: int | None
            Random seed for reproducibility. If None, no seed is set.

        Returns
        -------
        TensorLike
            Parameter samples (column order is given by self.param_names)
        """

        if random_seed is not None:
            set_random_seed(random_seed)  # type: ignore PGH003
        lhd = LatinHypercube(self.param_bounds)
        return lhd.sample(n_samples)

    @abstractmethod
    def _forward(self, x: TensorLike) -> TensorLike | None:
        """
        Abstract method to perform the forward simulation.

        Parameters
        ----------
        x: TensorLike
            Input parameters into the simulation forward run.

        Returns
        -------
        TensorLike | None
            Simulated output tensor. Shape = (1, self.out_dim).
            For example, if the simulator outputs two simulated variables,
            then the shape would be (1, 2). None if the simulation fails.
        """

    def forward(self, x: TensorLike) -> TensorLike | None:
        """
        Generate samples from input data using the simulator. Combines the
        abstract method `_forward` with some validation checks.

        Parameters
        ----------
        x: TensorLike
            Input tensor of shape (n_samples, self.in_dim).

        Returns
        -------
        TensorLike
            Simulated output tensor. None if the simulation failed.
        """
        y = self._forward(self.check_matrix(x))
        if isinstance(y, TensorLike):
            y = self.check_matrix(y)
            x, y = self.check_pair(x, y)
            return y
        return None

    def forward_batch(self, x: TensorLike) -> TensorLike:
        """Run multiple simulations with different parameters.

        For infallible simulators that always succeed.
        If your simulator might fail, use `forward_batch_skip_failures()` instead.

        Parameters
        ----------
        x: TensorLike
            Tensor of input parameters to make predictions for.

        Returns
        -------
        TensorLike
            Tensor of simulation results of shape (n_batch, self.out_dim).

        Raises
        ------
        RuntimeError
            If the number of simulations does not match the input.
            Use `forward_batch_skip_failures()` to handle failures.
        """
        results, x_valid = self.forward_batch_skip_failures(x)

        # Raise an error if the number of simulations does not match the input
        if x.shape[0] != x_valid.shape[0]:
            msg = (
                "Some simulations failed. Use forward_batch_skip_failures() to handle "
                "failures."
            )
            raise RuntimeError(msg)

        return results

    def forward_batch_skip_failures(
        self, x: TensorLike
    ) -> tuple[TensorLike, TensorLike]:
        """Run multiple simulations, skipping any that fail.

        For simulators where for some inputs the simulation can fail.
        Failed simulations are skipped, and only successful results are returned
        along with their corresponding input parameters.

        Parameters
        ----------
        x: TensorLike
            Tensor of input parameters to make predictions for.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            Tuple of (simulation_results, valid_input_parameters).
            Only successful simulations are included.
        """
        self.logger.info("Running batch simulation for %d samples", len(x))

        results = []
        successful = 0
        valid_idx = []

        # Process each sample with progress tracking
        for i in tqdm(
            range(len(x)),
            desc="Running simulations",
            disable=not self.progress_bar,
            total=len(x),
            unit="sample",
            unit_scale=True,
        ):
            logger.debug("Running simulation for sample %d/%d", i + 1, len(x))
            result = self.forward(x[i : i + 1])
            if result is not None:
                results.append(result)
                successful += 1
                valid_idx.append(i)
                logger.debug("Simulation %d/%d successful", i + 1, len(x))
            else:
                logger.warning(
                    "Simulation %d/%d failed. Result is None.", i + 1, len(x)
                )

        # Report results
        self.logger.info(
            "Successfully completed %d/%d simulations (%.1f%%)",
            successful,
            len(x),
            (successful / len(x) * 100 if len(x) > 0 else 0.0),
        )

        # stack results into a 2D array on first dim using torch
        results_tensor = torch.cat(results, dim=0)

        return results_tensor, x[valid_idx]

    def get_parameter_idx(self, name: str) -> int:
        """
        Get the index of a specific parameter.

        Parameters
        ----------
        name: str
            Name of the parameter to retrieve.

        Returns
        -------
        float
            Index of the specified parameter.
        """
        if name not in self._param_names:
            raise ValueError(f"Parameter {name} not found.")
        return self._param_names.index(name)
