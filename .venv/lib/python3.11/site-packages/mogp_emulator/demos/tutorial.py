import numpy as np
from mogp_emulator.demos.projectile import simulator, print_results
import mogp_emulator
import mogp_emulator.validation

try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

# An end-to-end tutorial illustrating model calibration using mogp_emulator

# First, we need to set up our experimental design. We would like our drag coefficient to be
# on a logarithmic scale and initial velocity to be on a linear scale. However, our simulator
# does the drag coefficient transformation for us, so we simply can specity the exponent on
# a linear scale.

# We will use a Latin Hypercube Design. To specify, we give the distribution that we would like
# the parameter to take. By default, we assume a uniform distribution between two endpoints, which
# we will use for this simulation.

# Once we construct the design, can draw a specified number of samples as shown.

lhd = mogp_emulator.LatinHypercubeDesign([(-5., 1.), (0., 1000.)])

n_simulations = 50
simulation_points = lhd.sample(n_simulations)
simulation_output = np.array([simulator(p) for p in simulation_points])

# Next, fit the surrogate GP model using MAP with the default priors
# Print out hyperparameter values as correlation lengths and sigma

gp = mogp_emulator.GaussianProcess(simulation_points, simulation_output, nugget="fit")
gp = mogp_emulator.fit_GP_MAP(gp, n_tries=1)

print("Correlation lengths = {}".format(gp.theta.corr))
print("Sigma = {}".format(np.sqrt(gp.theta.cov)))
print("Nugget = {}".format(np.sqrt(gp.theta.nugget)))

# Validate emulator by comparing to true simulated value and compute standard
# errors
# Errors are sorted with the largest variance first, idx values can be used to
# similarly sort inputs or prediction results

n_valid = 10
validation_points = lhd.sample(n_valid)
validation_output = np.array([simulator(p) for p in validation_points])

mean, var, _ = gp.predict(validation_points)

errors, idx = mogp_emulator.validation.standard_errors(gp, validation_points, validation_output)

print_results(validation_points[idx], errors, var[idx])

# Finally, perform history matching. Sample densely from the experimental design and
# determine which points are consistent with the data using the GP predictions
# We compute which points are "Not Ruled Out Yet" (NROY)

n_predict = 10000
prediction_points = lhd.sample(n_predict)

hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[2000., 400.])

nroy_points = hm.get_NROY()

print("Ruled out {} of {} points".format(n_predict - len(nroy_points), n_predict))

# If plotting enabled, visualize results

if makeplots:
    plt.figure()
    plt.plot(prediction_points[nroy_points,0], prediction_points[nroy_points,1], "o", label="NROY points")
    plt.plot(simulation_points[:,0], simulation_points[:,1],"o", label="Simulation Points")
    plt.plot(validation_points[:,0], validation_points[:,1],"o", label="Validation Points")
    plt.xlabel("log Drag Coefficient")
    plt.ylabel("Launch velocity (m/s)")
    plt.legend()
    plt.show()