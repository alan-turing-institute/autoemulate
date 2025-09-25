import logging
from abc import ABC, abstractmethod

import torch
from scipy.stats import qmc
from tqdm import tqdm

from autoemulate.core.logging_config import get_configured_logger
from autoemulate.core.types import TensorLike
from autoemulate.data.utils import ValidationMixin, set_random_seed

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
        Initialize the simulator with parameter ranges and output names.

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
        # separate out param bounds to sample from and constants
        self.sample_param_bounds = [b for b in self.param_bounds if b[0] != b[1]]
        self.constant_params = {
            idx: b[0] for idx, b in enumerate(self.param_bounds) if b[0] == b[1]
        }
        self._output_names = output_names
        self._in_dim = len(self.param_names)
        self._out_dim = len(self.output_names)
        self._has_sample_forward = False
        self.logger, self.progress_bar = get_configured_logger(log_level)

    @classmethod
    def simulator_name(cls) -> str:
        """Get the name of the simulator class."""
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
        self.sample_param_bounds = [b for b in self.param_bounds if b[0] != b[1]]
        self.constant_params = {
            idx: b[0] for idx, b in enumerate(self.param_bounds) if b[0] == b[1]
        }

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
        self, n_samples: int, random_seed: int | None = None, method: str = "lhs"
    ) -> TensorLike:
        """
        Generate random samples using Quasi-Monte Carlo methods.

        Available methods are Sobol or Latin Hypercube Sampling. For overview, see
        the scipy documentation:
        https://docs.scipy.org/doc/scipy/reference/stats.qmc.html

        Parameters
        ----------
        n_samples: int
            Number of samples to generate.
        random_seed: int | None
            Random seed for reproducibility. If None, no seed is set.
        method: str
            Sampling method to use. One of ["lhs", "sobol"].

        Returns
        -------
        TensorLike
            Parameter samples (column order is given by self.param_names)
        """
        if random_seed is not None:
            set_random_seed(random_seed)  # type: ignore PGH003

        if len(self.sample_param_bounds) == 0:
            # All parameters are constant - broadcast to n_samples
            const_vals = torch.tensor(list(self.constant_params.values()))
            return const_vals.repeat(n_samples, 1)

        if method.lower() == "lhs":
            sampler = qmc.LatinHypercube(d=len(self.sample_param_bounds))
        elif method.lower() == "sobol":
            sampler = qmc.Sobol(d=len(self.sample_param_bounds))
        else:
            msg = (
                f"Invalid sampling method: {method}. "
                "Supported methods are 'lhs' and 'sobol'."
            )
            raise ValueError(msg)

        # Samples are drawn from [0, 1]^d so need to scale them
        samples = sampler.random(n=n_samples)
        scaled_samples = qmc.scale(
            samples,
            [b[0] for b in self.sample_param_bounds],
            [b[1] for b in self.sample_param_bounds],
        )
        scaled_samples = torch.tensor(scaled_samples, dtype=torch.float32)

        # Insert constant parameters at correct indices
        full_samples = torch.empty((n_samples, self.in_dim), dtype=torch.float32)
        sample_idx = 0
        for idx in range(self.in_dim):
            if idx in self.constant_params:
                full_samples[:, idx] = self.constant_params[idx]
            else:
                full_samples[:, idx] = scaled_samples[:, sample_idx]
                sample_idx += 1

        return full_samples

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
            then the shape would be (1, 2).
        """

    def forward(self, x: TensorLike, allow_failures: bool = True) -> TensorLike | None:
        """
        Generate samples from input data using the simulator.

        Combines the abstract method `_forward` with some validation checks.
        If there is a failure during the forward pass of the simulation,
        the error is logged and None is returned.

        Parameters
        ----------
        x: TensorLike
            Input tensor of shape (n_samples, self.in_dim).
        allow_failures: bool
            Whether to allow failures during simulation.
            Default is True. When true, failed simulations will return None instead
            of raising an error. When False, error is raised.

        Returns
        -------
        TensorLike
            Simulated output tensor. None if the simulation failed.
        """
        try:
            y = self._forward(self.check_tensor_is_2d(x))
            if isinstance(y, TensorLike):
                x, y = self.check_pair(x, y)
                return y
        except Exception as e:
            if not allow_failures:
                self.logger.error("Error occurred during simulation: %s", e)
                raise
            self.logger.warning("Simulation failed with error %s. Returning None", e)
        return None

    def forward_batch(
        self, x: TensorLike, allow_failures: bool = True
    ) -> tuple[TensorLike, TensorLike]:
        """
        Run multiple simulations.

        If allow_failures is False, failed simulations will raise an error.
        Otherwise, failed simulations are skipped, and only successful results
        are returned along with their corresponding input parameters.

        Parameters
        ----------
        x: TensorLike
            Tensor of input parameters to make predictions for.
        allow_failures: bool
            Whether to allow failures during simulation.
            Default is True. When true, failed simulations will return None instead
            of raising an error. When False, error is raised.

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
            result = self.forward(x[i : i + 1], allow_failures=allow_failures)
            if result is not None:
                results.append(result)
                valid_idx.append(i)
                successful += 1
                logger.debug("Simulation %d/%d successful", i + 1, len(x))
            else:
                logger.warning(
                    "Simulation %d/%d failed. Result is None"
                    "and is not appended to the results",
                    i + 1,
                    len(x),
                )

        # Report results
        self.logger.info(
            "Successfully completed %d/%d simulations (%.1f%%)",
            successful,
            len(x),
            (successful / len(x) * 100 if len(x) > 0 else 0.0),
        )

        # handle no simulation results
        if results == []:
            return torch.empty((0, self.out_dim)), torch.empty((0, self.in_dim))

        # stack results into a 2D array on first dim using torch
        self.results_tensor = torch.cat(results, dim=0)

        return self.results_tensor, x[valid_idx]

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

    def get_outputs_as_dict(self) -> dict[str, TensorLike]:
        """
        Return simulation results as a dictionary with output names as keys.

        Returns
        -------
        dict[str, TensorLike]
            Dictionary where keys are output names and values are tensors
            of shape (n_samples,) for each output dimension.
        """
        # Create dictionary mapping output names to their corresponding columns
        output_dict = {}
        for i, output_name in enumerate(self.output_names):
            output_dict[output_name] = self.results_tensor[:, i]

        return output_dict
