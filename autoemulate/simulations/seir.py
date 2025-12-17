from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from autoemulate.core.types import NumpyLike, TensorLike
from autoemulate.simulations.base import Simulator


def simulate_seir_epidemic(
    x: NumpyLike,
    N: int = 1000,
    I0: int = 1,
    E0: int = 0,
) -> float:
    """
    Simulate an epidemic using the SEIR model.

    Parameters
    ----------
    x : NumpyLike
        SEIR parameters [beta, gamma, sigma].
    N : int
        Total population.
    I0 : int
        Initial infected.
    E0 : int
        Initial exposed.

    Returns
    -------
    peak_infection_rate : float
        Peak infection fraction I_max / N.
    """
    if len(x) != 3:
        raise ValueError(f"Expected 3 parameters [beta, gamma, sigma], got {len(x)}")

    beta, gamma, sigma = x

    S0 = N - I0 - E0
    R0 = 0
    t_span = (0.0, 160.0)
    y0 = [S0, E0, I0, R0]

    def seir_model(t, y, N, beta, gamma, sigma):
        S, E, I, R = y  # noqa: E741
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    t_eval = np.linspace(t_span[0], t_span[1], 160)
    sol = solve_ivp(
        seir_model,
        t_span,
        y0,
        args=(N, beta, gamma, sigma),
        t_eval=t_eval,
        vectorized=False,
    )

    _, E, I, R = sol.y  # noqa: E741 
    I_max = np.max(I)

    return float(I_max) / float(N)


class SEIRSimulator(Simulator):
    """Simulator of infectious disease spread using the SEIR model."""

    def __init__(
        self,
        parameters_range=None,
        output_names=None,
        log_level: str = "progress_bar",
    ):
        if parameters_range is None:
            parameters_range = {
                "beta": (0.1, 0.5),
                "gamma": (0.01, 0.2),
                "sigma": (0.05, 0.3),
            }
        if output_names is None:
            output_names = ["infection_rate"]

        super().__init__(parameters_range, output_names, log_level)

    def _forward(self, x: TensorLike) -> TensorLike:
        """
        Simulate the epidemic using the SEIR model.

        Parameters
        ----------
        x : TensorLike
            Input parameter values [beta, gamma, sigma].

        Returns
        -------
        TensorLike
            Peak infection rate (fraction of population).
        """
        if x.shape[0] != 1:
            raise ValueError(
                f"SEIRSimulator._forward expects a single input, got {x.shape[0]}"
            )

        y = simulate_seir_epidemic(x.cpu().numpy()[0])
        return torch.tensor([y], dtype=torch.float32).view(-1, 1)
