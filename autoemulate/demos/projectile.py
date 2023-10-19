# from https://github.com/alan-turing-institute/mogp-emulator/blob/main/mogp_emulator/demos/projectile.py

import numpy as np
import scipy
from scipy.integrate import solve_ivp

# Create our simulator, which solves a nonlinear differential equation describing projectile
# motion with drag. A projectile is launched from an initial height of 2 meters at an
# angle of 45 degrees and falls under the influence of gravity and air resistance.
# Drag is proportional to the square of the velocity. We would like to determine the distance
# travelled by the projectile as a function of the drag coefficient and the launch velocity.

# define functions needed for simulator

def f(t, y, c):
    "Compute RHS of system of differential equations, returning vector derivative"

    # check inputs and extract

    assert len(y) == 4
    assert c >= 0.

    vx = y[0]
    vy = y[1]

    # calculate derivatives

    dydt = np.zeros(4)

    dydt[0] = -c*vx*np.sqrt(vx**2 + vy**2)
    dydt[1] = -9.8 - c*vy*np.sqrt(vx**2 + vy**2)
    dydt[2] = vx
    dydt[3] = vy

    return dydt

def event(t, y, c):
    "event to trigger end of integration"

    assert len(y) == 4
    assert c >= 0.

    return y[3]

event.terminal = True

# now can define simulator

def simulator_base(x):
    "simulator to solve ODE system for projectile motion with drag. returns distance projectile travels"

    # unpack values

    assert len(x) == 2
    assert x[1] > 0.

    c = 10.**x[0]
    v0 = x[1]

    # set initial conditions

    y0 = np.zeros(4)

    y0[0] = v0/np.sqrt(2.)
    y0[1] = v0/np.sqrt(2.)
    y0[3] = 2.

    # run simulation

    results = solve_ivp(f, (0., 1.e8), y0, events=event, args = (c,))

    return results

def simulator(x):
    "simulator to solve ODE system for projectile motion with drag. returns distance projectile travels"

    results = simulator_base(x)

    return results.y_events[0][0][2]

def simulator_multioutput(x):
    "simulator to solve ODE system with multiple outputs"
    
    results = simulator_base(x)
    
    return (results.y_events[0][0][2],
            np.sqrt(results.y_events[0][0][0]**2 + results.y_events[0][0][1]**2))

# functions for printing out results

def print_results(inputs, arg, var):
    "convenience function for printing out generic results"

    print("---------------------------------------------------------------------------------")

    for pp, m, v in zip(inputs, arg, var):
        print("{}      {}       {}".format(pp, m, v))

def print_predictions(inputs, pred, var):
    "convenience function for printing predictions"
    
    print("Target Point                Predicted Mean             Predictive Variance")
    print_results(inputs, pred, var)

def print_errors(inputs, errors, var):
    "convenience function for printing out results and computing mean square error"

    print("Target Point                Standard Error             Predictive Variance")
    print_results(inputs, errors, var)
    print("Mean squared error: {}".format(np.sum(errors**2)/len(errors)))