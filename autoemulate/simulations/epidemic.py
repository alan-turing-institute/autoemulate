import numpy as np
from scipy.integrate import solve_ivp


def simulate_epidemic(x, N=1000, I0=1):
    """Simulate an epidemic using the SIR model.

    Parameters
    ----------
    x : array-like
        The parameters of the SIR model. The first element is the transmission rate (beta) and the second element is the recovery rate (gamma).
    N : int, optional
        The total population size.
    I0 : int, optional
        The initial number of infected individuals.

    Returns
    -------
    peak_infection_rate : float
        The peak infection rate as a fraction of the total population.
    """

    # check inputs
    assert len(x) == 2
    assert N > 0
    assert I0 > 0 and I0 < N
    assert x[0] > 0
    assert x[1] > 0

    # unpack parameters
    beta = x[0]
    gamma = x[1]

    S0 = N - I0
    R0 = 0
    t_span = [0, 160]
    y0 = [S0, I0, R0]

    def sir_model(t, y, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    t_eval = np.linspace(
        t_span[0], t_span[1], 160
    )  # Evaluate each day within the time span
    sol = solve_ivp(sir_model, t_span, y0, args=(N, beta, gamma), t_eval=t_eval)

    S, I, R = sol.y
    I_max = np.max(I)
    peak_infection_rate = I_max / N

    return peak_infection_rate
