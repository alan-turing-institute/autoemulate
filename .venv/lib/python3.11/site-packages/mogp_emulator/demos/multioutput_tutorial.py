import numpy as np
from mogp_emulator.demos.projectile import simulator_multioutput, print_errors
import mogp_emulator
import mogp_emulator.validation

try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

# An end-to-end tutorial illustrating model calibration with multiple outputs using mogp_emulator

# First, we need to set up our experimental design. We would like our drag coefficient to be
# on a logarithmic scale and initial velocity to be on a linear scale. However, our simulator
# does the drag coefficient transformation for us, so we simply can specity the exponent on
# a linear scale.

# We will use a Latin Hypercube Design. To specify, we give the distribution that we would like
# the parameter to take. By default, we assume a uniform distribution between two endpoints, which
# we will use for this simulation.

# Once we construct the design, can draw a specified number of samples as shown.

if __name__ == "__main__": # this is required for multiprocessing to work correctly!

    lhd = mogp_emulator.LatinHypercubeDesign([(-5., 1.), (0., 1000.)])

    n_simulations = 50
    simulation_points = lhd.sample(n_simulations)

# Run simulator. For the multioutput simulator, returns (distance, velocity) pair

    simulation_output = np.array([simulator_multioutput(p) for p in simulation_points]).T

# Next, fit the surrogate MOGP model using MAP with the default priors
# Print out hyperparameter values as correlation lengths and sigma

    gp = mogp_emulator.MultiOutputGP(simulation_points, simulation_output, nugget="fit")
    gp = mogp_emulator.fit_GP_MAP(gp, n_tries=2)

    print("Correlation lengths (distance)= {}".format(gp.emulators[0].theta.corr))
    print("Correlation lengths (velocity)= {}".format(gp.emulators[1].theta.corr))
    print("Sigma (distance)= {}".format(np.sqrt(gp.emulators[0].theta.cov)))
    print("Sigma (velocity)= {}".format(np.sqrt(gp.emulators[1].theta.cov)))
    print("Nugget (distance)= {}".format(np.sqrt(gp.emulators[0].theta.nugget)))
    print("Nugget (velocity)= {}".format(np.sqrt(gp.emulators[1].theta.nugget)))

# Validate emulators by comparing to true simulated values

    n_valid = 10
    validation_points = lhd.sample(n_valid)
    validation_output = np.array([simulator_multioutput(p) for p in validation_points]).T

    mean, var, _ = gp.predict(validation_points)

    errors = mogp_emulator.validation.standard_errors(gp, validation_points, validation_output)

    for errval, v in zip(errors, var):
        e, idx = errval
        print_errors(validation_points[idx], e[idx], v[idx])

# Finally, perform history matching. Sample densely from the experimental design and
# determine which points are consistent with the data using the GP predictions
# We compute which points are "Not Ruled Out Yet" (NROY)

# Note that our observations are now vectors, with the same ordering as the
# simulation output

    n_predict = 10000
    prediction_points = lhd.sample(n_predict)

    hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[np.array([2000., 100.]),
                                                                             np.array([100., 5.])])

    nroy_points = hm.get_NROY(rank=0)

    print("Ruled out {} of {} points".format(n_predict - len(nroy_points), n_predict))

# If plotting enabled, visualize results

    if makeplots:
        plt.figure()
        plt.plot(prediction_points[nroy_points,0], prediction_points[nroy_points,1], "o", label="NROY points")
        plt.plot(simulation_points[:,0], simulation_points[:,1],"o", label="Simulation Points")
        plt.xlabel("log Drag Coefficient")
        plt.ylabel("Launch velocity (m/s)")
        plt.legend()
        plt.show()