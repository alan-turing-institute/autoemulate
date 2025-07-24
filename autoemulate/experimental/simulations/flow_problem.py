from collections.abc import Callable

import numpy as np
import scipy as sp
import torch

from autoemulate.experimental.simulations.base import Simulator
from autoemulate.experimental.types import TensorLike


class FlowProblem(Simulator):
    """
    The system simulated is a tube with an input flow rate at any given time.
    The tube is divided to `ncomp` compartments which allows for the study of
    the pressure and flow rate at various points in the tube.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        log_level: str = "progress_bar",
        ncycles: int = 10,
        ncomp: int = 10,
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
        ncycles: int
            Number of cycles to simulate.
        ncomp: int
            Number of compartments in the tube.
        """
        if parameters_range is None:
            parameters_range = {
                # Cardiac cycle period (s)
                "T": (0.5, 2.0),
                # Pulse duration (s)
                "td": (0.1, 0.5),
                # Amplitude (e.g., pressure or flow rate)
                "amp": (100.0, 1000.0),
                # Time step (s)
                "dt": (0.0001, 0.01),
                # Compliance (unit varies based on context)
                "C": (20.0, 60.0),
                # Resistance (unit varies based on context)
                "R": (0.01, 0.1),
                # Inductance (unit varies based on context)
                "L": (0.001, 0.005),
                # Outflow resistance (unit varies based on context)
                "R_o": (0.01, 0.05),
                # Initial pressure (unit varies based on context)
                "p_o": (5.0, 15.0),
            }
        if output_names is None:
            output_names = ["pressure"]
        super().__init__(parameters_range, output_names, log_level)
        self.ncycles = ncycles
        self.ncomp = ncomp

    def _forward(self, x: TensorLike) -> TensorLike:
        """
        Calculate the pressure and flow rate in the tube compartments.

        Parameters
        ----------
            x: TensorLike
                Tensor of input parameter values to simulate. Will be converted to a
                dictionary internally.
                    - `T` (float): cycle length
                    - `td` (float): pulse duration, make sure to make this less than T
                    - `amp` (float): inflow amplitude
                    - `dt` (float): temporal discretisation resolution
                    - `C` (float): tube average compliance
                    - `R` (float): tube average impedance
                    - `L` (float): hydraulic impedance, inertia
                    - `R_o` (float) : outflow resistance
                    - `p_o` (float) : outflow pressure

        Returns
        -------
            TensorLike
                Presssure in the tube compartments at the end of the simulation.
        """

        # Convert tensor input to dictionary
        x_values = x.squeeze().tolist() if hasattr(x, "squeeze") else x

        # Create parameter dictionary
        params = {}
        for i, param_name in enumerate(self.param_names):
            value = x_values[i]
            params[param_name] = value

        def dfdt_fd(params_dict, t: float, y: np.ndarray, Q_in):
            Cn = params_dict["C"] / self.ncomp
            Rn = params_dict["R"] / self.ncomp
            Ln = params_dict["L"] / self.ncomp

            out = np.zeros((self.ncomp, 2))
            y_temp = y.reshape((-1, 2))

            for i in range(self.ncomp):
                if i > 0:
                    out[i, 0] = (y_temp[i - 1, 1] - y_temp[i, 1]) / Cn
                else:
                    out[i, 0] = (Q_in(t % params_dict["T"]) - y_temp[i, 1]) / Cn
                if i < self.ncomp - 1:
                    out[i, 1] = (
                        -y_temp[i + 1, 0] + y_temp[i, 0] - Rn * y_temp[i, 1]
                    ) / Ln
                else:
                    out[i, 1] = (
                        -params_dict["p_o"]
                        + y_temp[i, 0]
                        - (Rn + params_dict["R_o"]) * y_temp[i, 1]
                    ) / Ln
            return out.reshape((-1,))

        def generate_pulse_function(params_dict: dict) -> Callable:
            def Q_mi_lambda(t):
                return (
                    np.sin(np.pi / params_dict["td"] * t) ** 2.0
                    * np.heaviside(params_dict["td"] - t, 0.0)
                    * params_dict["amp"]
                )

            return Q_mi_lambda

        def dfdt_fd_spec(t, y):
            return dfdt_fd(
                params_dict=params, t=t, y=y, Q_in=generate_pulse_function(params)
            )

        res = sp.integrate.solve_ivp(
            dfdt_fd_spec,
            [0.0, params["T"] * self.ncycles],
            y0=np.zeros(self.ncomp * 2),
            method="BDF",
            max_step=params["dt"],
        )
        res.y = res.y[:, res.t >= params["T"] * (self.ncycles - 1)]
        # Get the peak pressure across the
        # first ncomp compartments across all time steps
        peak_pressure = res.y[: self.ncomp, :].max()
        return torch.tensor([[peak_pressure]], dtype=torch.float32)
