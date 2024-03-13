import numpy as np
from scipy.integrate import solve_ivp


def sir_model(t, y, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def simulate_epidemic(X):
    """
    A simplified SIR epidemic model.

    Parameters:
    N (int): Total population.
    I0 (int): Initial number of infected individuals.
    beta (float): Transmission rate.
    gamma (float): Recovery rate.

    Returns:
    Peak infection rate (float): Maximum proportion of the population infected at any time.
    """
    N, I0, beta, gamma = X[0], X[1], X[2], X[3]
    S0 = N - I0
    R0 = 0
    t_span = [0, 160]  # Time span for the simulation in days
    y0 = [S0, I0, R0]  # Initial conditions

    sol = solve_ivp(sir_model, t_span, y0, args=(N, beta, gamma), dense_output=True)

    I_max = max(sol.sol(t_span)[1])
    peak_infection_rate = I_max / N

    return peak_infection_rate
