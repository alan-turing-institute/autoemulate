import numpy as np
from projectile import simulator
import mogp_emulator

try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

# define a helper function for making plots

def plot_solution(field, title, filename, simulation_points, validation_points, tri):
    plt.figure(figsize=(4,3))
    plt.tripcolor(validation_points[:,0], validation_points[:,1], tri.triangles,
                  field, vmin=0, vmax=5000.)
    cb = plt.colorbar()
    plt.scatter(simulation_points[:,0], simulation_points[:,1])
    plt.xlabel("log drag coefficient")
    plt.ylabel("Launch velocity (m/s)")
    cb.set_label("Projectile distance (m)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    
# A tutorial illustrating effectiveness of mean functions and priors for GP emulation

# Most often, we are not able to sample very densely from a simulation, so we
# have relatively few samples per input parameter. This can lead to some problems
# when constructing a robust emulator. This tutorial illustrates how we can build
# better emulators using the tools in mogp_emulator.

# We need to draw some samples from the space to run some simulations and build our
# emulators. We use a LHD design with only 6 sample points.

lhd = mogp_emulator.LatinHypercubeDesign([(-4., 0.), (0., 1000.)])

n_simulations = 6
simulation_points = lhd.sample(n_simulations)
simulation_output = np.array([simulator(p) for p in simulation_points])

# Next, fit the surrogate GP model using MLE, zero mean, and no priors.
# Print out hyperparameter values as correlation lengths, sigma, and nugget

# Note that as of v0.6.0, you have to explicitly choose weak priors (if none are
# provided then the GP tries to fit some for you based on your data)

priors = mogp_emulator.Priors.GPPriors(n_corr=2, nugget_type="adaptive")

gp = mogp_emulator.GaussianProcess(simulation_points, simulation_output, priors=priors)
gp = mogp_emulator.fit_GP_MAP(gp)

print("Zero mean and no priors:")
print("Correlation lengths = {}".format(gp.theta.corr))
print("Covariance scale (sigma^2)= {}".format(gp.theta.cov))
print("Nugget = {}".format(gp.nugget))
print()

# We can look at how the emulator performs by comparing the emulator output to
# a large number of validation points. Since this simulation is cheap, we can
# actually compute this for a large number of points.

n_valid = 1000
validation_points = lhd.sample(n_valid)
validation_output = np.array([simulator(p) for p in validation_points])

if makeplots:
    import matplotlib.tri
    tri = matplotlib.tri.Triangulation((validation_points[:,0]+4.)/4.,
                                       (validation_points[:,1]/1000.))

    plot_solution(validation_output, "True simulator", "simulator_output.png",
                  simulation_points, validation_points, tri)

# Now predict values with the emulator and plot output and error

predictions = gp.predict(validation_points)

if makeplots:
    plot_solution(predictions.mean, "MLE emulator", "emulator_output_MLE.png",
                  simulation_points, validation_points, tri)

# This is not very good! The simulation points are too sparsely sampled to give the
# emulator any idea what to do about the function shape. We just know the value at a few
# points, and it throws up its hands and predicts zero everywhere else.

# To improve this, we will specify a mean function and some priors to ensure that if we are
# far away from an evaluation point we will still get some information from the emulator.

# We specify the mean function using a string, which follows a similar approach to R-style
# formulas. There is an implicit constant term, and we use x[index] to specify how we
# want the formula to depend on the inputs. We choose a simple linear form here, which has
# three fitting parameters in addition to the correlations lengths, sigma, and nugget
# parameters above.

meanfunc = "x[0]+x[1]"

# We now set priors for all of the hyperparameters to better constrain the estimation procedure.
# We assume normal priors for the mean function parameters with a large variance (to not constrain
# our choice too much). Note that the mean function parameters are on a linear scale, while the
# correlation lengths, sigma, and nugget are on a logarithmic scale. Thus, if we choose normal
# priors on the correlation lengths, these will actually be lognormal distributions.

# Finally, we choose inverse gamma and gamma distributions for the priors on sigma and the nugget
# as those are typical conjugate priors for variances/precisions. We pick them to be where they are as
# we expect sigma to be large (as the function is very sensitive to inputs) while we want the
# nugget to be small.

priors = mogp_emulator.Priors.GPPriors(mean=mogp_emulator.Priors.MeanPriors(mean=np.zeros(3), cov=10.),
                                       corr=[mogp_emulator.Priors.LogNormalPrior(1., 1.),
                                             mogp_emulator.Priors.LogNormalPrior(1., 1.)],
                                       cov=mogp_emulator.Priors.InvGammaPrior(1., 1.),
                                       nugget=mogp_emulator.Priors.GammaPrior(1., 1.))

# Now, construct another GP using the mean function and priors. note that we also specify that we
# want to estimate the nugget based on our prior, rather than adaptively fitting it as we did in
# the first go.

gp_map = mogp_emulator.GaussianProcess(simulation_points, simulation_output,
                                       mean=meanfunc, priors=priors, nugget="fit")
gp_map = mogp_emulator.fit_GP_MAP(gp_map)

print("With mean and priors:")
print("Mean function parameters = {}".format(gp_map.theta.mean))
print("Correlation lengths = {}".format(gp_map.theta.corr))
print("Covariance Scale (sigma^2) = {}".format(gp_map.theta.cov))
print("Nugget = {}".format(gp_map.nugget))

# Use the new fit GP to predict the validation points and plot to see if this improved
# the fit to the true data:

predictions_map = gp_map.predict(validation_points)

if makeplots:
    plot_solution(predictions_map.mean, "Mean/Prior emulator", "emulator_output_MAP.png",
                  simulation_points, validation_points, tri)
