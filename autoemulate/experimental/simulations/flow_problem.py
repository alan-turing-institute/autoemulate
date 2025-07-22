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
                Dictionary of input parameter values to simulate:
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

        def dfdt_fd(x, t: float, y: np.ndarray, Q_in):
            Cn = x["C"] / x["ncomp"]
            Rn = x["R"] / x["ncomp"]
            Ln = x["L"] / x["ncomp"]

            out = np.zeros((x["ncomp"], 2))
            y_temp = y.reshape((-1, 2))

            for i in range(x["ncomp"]):
                if i > 0:
                    out[i, 0] = (y_temp[i - 1, 1] - y_temp[i, 1]) / Cn
                else:
                    out[i, 0] = (Q_in(t % x["T"]) - y_temp[i, 1]) / Cn
                if i < x["ncomp"] - 1:
                    out[i, 1] = (
                        -y_temp[i + 1, 0] + y_temp[i, 0] - Rn * y_temp[i, 1]
                    ) / Ln
                else:
                    out[i, 1] = (
                        -x["p_o"] + y_temp[i, 0] - (Rn + x["R_o"]) * y_temp[i, 1]
                    ) / Ln
            return out.reshape((-1,))

        def generate_pulse_function(x: TensorLike) -> callable:  # type: ignore  # noqa: PGH003
            Q_mi_lambda = (  # noqa: E731
                lambda t: np.sin(np.pi / x["td"] * t) ** 2.0  # type: ignore  # noqa: PGH003
                * np.heaviside(x["td"] - t, 0.0)  # type: ignore  # noqa: PGH003
                * x["amp"]  # type: ignore  # noqa: PGH003
            )
            return Q_mi_lambda  # noqa: RET504

        dfdt_fd_spec = lambda t, y: dfdt_fd(t=t, y=y, Q_in=generate_pulse_function(x))  # type: ignore  # noqa: E731, PGH003
        res = sp.integrate.solve_ivp(
            dfdt_fd_spec,
            [0.0, x["T"] * x["ncycles"]],  # type: ignore  # noqa: PGH003
            y0=np.zeros(x["ncomp"] * 2),  # type: ignore  # noqa: PGH003
            method="BDF",
            max_step=x["dt"],  # type: ignore  # noqa: PGH003
        )
        res.y = res.y[:, res.t >= x["T"] * (x["ncycles"] - 1)]  # type: ignore  # noqa: PGH003
        # res.t = res.t[res.t >= x['T'] * (x['ncycles'] - 1)]
        res.y = torch.tensor(res.y, dtype=torch.float32)  # type: ignore  # noqa: PGH003
        # res.t = torch.tensor(res.t, dtype=torch.float32)
        return res.y
