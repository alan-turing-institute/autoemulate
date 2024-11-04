# from https://github.com/alan-turing-institute/mogp-emulator/blob/main/mogp_emulator/demos/projectile.py
import numpy as np
from scipy.integrate import solve_ivp

# Create our simulator, which solves a nonlinear differential equation describing projectile
# motion with drag. A projectile is launched from an initial height of 2 meters at an
# angle of 45 degrees and falls under the influence of gravity and air resistance.
# Drag is proportional to the square of the velocity. We would like to determine the distance
# travelled by the projectile as a function of the drag coefficient and the launch velocity.

# define functions needed for simulator


def f(t, y, c):
    """
    Compute RHS of system of differential equations, returning vector derivative

    Parameters
    ----------
    t : float
        Time variable (not used).
    y : array
        Array of dependent variables (vx, vy, x, y).
    c : float
        Drag coefficient (non-negative).
    """

    # check inputs and extract

    assert len(y) == 4
    assert c >= 0.0

    vx = y[0]
    vy = y[1]

    # calculate derivatives

    dydt = np.zeros(4)

    dydt[0] = -c * vx * np.sqrt(vx**2 + vy**2)
    dydt[1] = -9.8 - c * vy * np.sqrt(vx**2 + vy**2)
    dydt[2] = vx
    dydt[3] = vy

    return dydt


def event(t, y, c):
    """
    Event to trigger end of integration. Stops when projectile hits ground.

    Parameters
    ----------
    t : float
        Time variable (not used).
    y : array
        Array of dependent variables (vx, vy, x, y).
    c : float
        Drag coefficient (non-negative).
    """

    assert len(y) == 4
    assert c >= 0.0

    return y[3]


event.terminal = True

# now can define simulator


def simulator_base(x):
    """
    Simulator to solve ODE system for projectile motion with drag. Returns distance projectile travels.

    Parameters
    ----------
    x : array
        Array of input parameters (c, v0).

    Returns
    -------
    results : scipy.integrate.OdeResult
        Results of ODE integration.
    """

    # unpack values

    assert len(x) == 2
    assert x[1] > 0.0

    c = 10.0 ** x[0]
    v0 = x[1]

    # set initial conditions

    y0 = np.zeros(4)

    y0[0] = v0 / np.sqrt(2.0)
    y0[1] = v0 / np.sqrt(2.0)
    y0[3] = 2.0

    # run simulation

    results = solve_ivp(f, (0.0, 1.0e8), y0, events=event, args=(c,))

    return results


def simulate_projectile(x):
    """
    Simulator to solve ODE system for projectile motion with drag. Returns distance projectile travels.

    Parameters
    ----------
    x : array
        Array of input parameters (c, v0).

    Returns
    -------
    distance : float
        Distance travelled by projectile.
    """

    results = simulator_base(x)

    return results.y_events[0][0][2]


def simulate_projectile_multioutput(x):
    """
    Simulator to solve ODE system with multiple outputs.

    Parameters
    ----------
    x : array
        Array of input parameters (c, v0).

    Returns
    -------
    distance : float
        Distance travelled by projectile.
    velocity : float
        Velocity of projectile at impact.
    """

    results = simulator_base(x)

    return (
        results.y_events[0][0][2],
        np.sqrt(results.y_events[0][0][0] ** 2 + results.y_events[0][0][1] ** 2),
    )


# functions for printing out results


def print_results(inputs, arg, var):
    """
    Convenience function for printing out generic results.

    Parameters
    ----------
    inputs : array
        Array of input values.
    arg : array
        Array of mean values.
    var : array
        Array of variance values.

    Returns
    -------
    None.
    """

    print(
        "---------------------------------------------------------------------------------"
    )

    for pp, m, v in zip(inputs, arg, var):
        print("{}      {}       {}".format(pp, m, v))


def print_predictions(inputs, pred, var):
    """
    Convenience function for printing predictions.

    Parameters
    ----------
    inputs : array
        Array of input values.
    pred : array
        Array of mean values.
    var : array
        Array of variance values.
    """

    print("Target Point                Predicted Mean             Predictive Variance")
    print_results(inputs, pred, var)


def print_errors(inputs, errors, var):
    """
    Convenience function for printing out results and computing mean square error.

    Parameters
    ----------
    inputs : array
        Array of input values.
    errors : array
        Array of errors.
    var : array
        Array of variance values.

    Returns
    -------
    None.
    """

    print("Target Point                Standard Error             Predictive Variance")
    print_results(inputs, errors, var)
    print("Mean squared error: {}".format(np.sum(errors**2) / len(errors)))
