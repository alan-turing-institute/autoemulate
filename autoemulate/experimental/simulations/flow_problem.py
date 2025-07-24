from collections.abc import Callable

import numpy as np
import scipy as sp
import torch

from autoemulate.experimental.simulations.base import Simulator
from autoemulate.experimental.types import TensorLike


class FlowProblem(Simulator):
    """
    The system simulated is a tube with an input flow rate at any given time.
    The tube is divided to 10 compartments which allows for the study of the pressure
    and flow rate at various points in the tube.
    """

    def __init__(
        self,
        param_ranges,
        output_names,
    ):
        super().__init__(param_ranges, output_names)

    def _forward(self, x: TensorLike) -> TensorLike:
        """
        Calculate the pressure and flow rate in the tube compartments.

        Parameters:
            x: TensorLike
                Tensor of input parameter values to simulate. Will be converted to a
                dictionary internally.
                    - `T` (float): cycle length
                    - `td` (float): pulse duration, make sure to make this less than T
                    - `amp` (float): inflow amplitude
                    - `dt` (float): temporal discretisation resolution
                    - `ncycles` (int): number of cycles to simulate
                    - `ncomp` (int): number of compartments in the tube
                    - `C` (float): tube average compliance
                    - `R` (float): tube average impedance
                    - `L` (float): hydraulic impedance, inertia
                    - `R_o` (float) : outflow resistance
                    - `p_o` (float) : outflow pressure

        Returns:
            TensorLike
                Presssure in the tube compartments at the end of the simulation.
        """

        # Convert tensor input to dictionary
        x_values = x.squeeze().tolist() if hasattr(x, "squeeze") else x

        # Create parameter dictionary
        params = {}
        for i, param_name in enumerate(self.param_names):
            value = x_values[i]
            # Convert integer parameters to int type
            if param_name in ["ncycles", "ncomp"]:
                params[param_name] = int(value)
            else:
                params[param_name] = value

        def dfdt_fd(params_dict, t: float, y: np.ndarray, Q_in):
            Cn = params_dict["C"] / params_dict["ncomp"]
            Rn = params_dict["R"] / params_dict["ncomp"]
            Ln = params_dict["L"] / params_dict["ncomp"]

            out = np.zeros((params_dict["ncomp"], 2))
            y_temp = y.reshape((-1, 2))

            for i in range(params_dict["ncomp"]):
                if i > 0:
                    out[i, 0] = (y_temp[i - 1, 1] - y_temp[i, 1]) / Cn
                else:
                    out[i, 0] = (Q_in(t % params_dict["T"]) - y_temp[i, 1]) / Cn
                if i < params_dict["ncomp"] - 1:
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
            Q_mi_lambda = (  # noqa: E731
                lambda t: np.sin(np.pi / params_dict["td"] * t) ** 2.0  # type: ignore  # noqa: PGH003
                * np.heaviside(params_dict["td"] - t, 0.0)  # type: ignore  # noqa: PGH003
                * params_dict["amp"]  # type: ignore  # noqa: PGH003
            )
            return Q_mi_lambda  # noqa: RET504

        dfdt_fd_spec = lambda t, y: dfdt_fd(  # noqa: E731
            params_dict=params, t=t, y=y, Q_in=generate_pulse_function(params)
        )  # type: ignore  # noqa: PGH003
        res = sp.integrate.solve_ivp(
            dfdt_fd_spec,
            [0.0, params["T"] * params["ncycles"]],  # type: ignore  # noqa: PGH003
            y0=np.zeros(params["ncomp"] * 2),  # type: ignore  # noqa: PGH003
            method="BDF",
            max_step=params["dt"],  # type: ignore  # noqa: PGH003
        )
        res.y = res.y[:, res.t >= params["T"] * (params["ncycles"] - 1)]  # type: ignore  # noqa: PGH003
        # res.t = res.t[res.t >= x['T'] * (x['ncycles'] - 1)]
        res.y = torch.tensor(res.y, dtype=torch.float32)  # type: ignore  # noqa: PGH003
        # res.t = torch.tensor(res.t, dtype=torch.float32)

        # Return pressure values from the last time step,
        # reshaped to (1, n_compartments*2)
        final_values = res.y[:, -1]  # Get final time step values
        return final_values.unsqueeze(0)  # Add batch dimension to match expected shape
