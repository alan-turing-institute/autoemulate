import numpy as np
import torch
from scipy.integrate import solve_ivp

from autoemulate.core.types import NumpyLike, TensorLike
from autoemulate.simulations.base import Simulator


class Epidemic(Simulator):
    """Simulator of infectious disease spread (SIR)."""

    def __init__(
        self,
        parameters_range=None,
        output_names=None,
        log_level: str = "progress_bar",
    ):
        if parameters_range is None:
            parameters_range = {"beta": (0.1, 0.5), "gamma": (0.01, 0.2)}
        if output_names is None:
            output_names = ["infection_rate"]
        super().__init__(parameters_range, output_names, log_level)

    def _forward(self, x: TensorLike) -> TensorLike:
        """
        Simulate the epidemic using the SIR model.

        Parameters
        ----------
        x: TensorLike
            input parameter values to simulate [beta, gamma]:
            - `beta`: the transimission rate per day
            - `gamma`: the recovery rate per day

        Returns
        -------
        TensorLike
            Peak infection rate.
        """
        assert x.shape[0] == 1, (
            f"Simulator._forward expects a single input, got {x.shape[0]}"
        )
        y = simulate_epidemic(x.cpu().numpy()[0])
        # TODO (#537): update with default dtype
        return torch.tensor([y], dtype=torch.float32).view(-1, 1)


def simulate_epidemic(x: NumpyLike, N: int = 1000, I0: int = 1) -> float:
    """
    Simulate an epidemic using the SIR model.

    Parameters
    ----------
    x: NumpyLike
        The parameters of the SIR model. The first element is the transmission rate
        (beta) and the second element is the recovery rate (gamma).
    N: int
        The total population size. Defaults to 1000.
    I0: int
        The initial number of infected individuals. Defaults to 1.

    Returns
    -------
    peak_infection_rate: float
        The peak infection rate as a fraction of the total population.
    """
    # check inputs
    assert len(x) == 2
    assert N > 0
    assert I0 > 0
    assert I0 < N
    assert x[0] > 0
    assert x[1] > 0

    # unpack parameters
    beta = x[0]
    gamma = x[1]

    S0 = N - I0
    R0 = 0
    t_span = [0, 160]
    y0 = [S0, I0, R0]

    def sir_model(t, y, N, beta, gamma):  # noqa: ARG001
        S, I, R = y  # noqa: E741
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    t_eval = np.linspace(
        t_span[0], t_span[1], 160
    )  # Evaluate each day within the time span
    sol = solve_ivp(sir_model, t_span, y0, args=(N, beta, gamma), t_eval=t_eval)

    S, I, R = sol.y  # noqa: E741
    I_max = np.max(I)

    # peak infection rate
    return I_max / N
