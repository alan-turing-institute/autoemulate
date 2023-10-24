import numpy as np
import mogp_emulator
from mogp_emulator.Kernel import UniformSqExp
from mogp_emulator.demos.projectile import print_predictions

# additional GP examples using different Kernels

# define some common variables

n_samples = 20
n_preds = 10

# define target function

def f(x):
    return 4.*np.exp(-0.5*((x[0] - 2.)**2/2. + (x[1] - 4.)**2/0.25))

# Experimental design -- requires a list of parameter bounds if you would like to use
# uniform distributions. If you want to use different distributions, you
# can use any of the standard distributions available in scipy to create
# the appropriate ppf function (the inverse of the cumulative distribution).
# Internally, the code creates the design on the unit hypercube and then uses
# the distribution to map from [0,1] to the real parameter space.

ed = mogp_emulator.LatinHypercubeDesign([(0., 5.), (0., 5.)])

# sample space

inputs = ed.sample(n_samples)

# run simulation

targets = np.array([f(p) for p in inputs])

###################################################################################

# First example -- standard Squared Exponential Kernel

print("Example 1: Squared Exponential")

# create GP and then fit using MLE

gp = mogp_emulator.GaussianProcess(inputs, targets)

gp = mogp_emulator.fit_GP_MAP(gp)

# look at hyperparameters (correlation lengths, covariance, and nugget)

print("Correlation lengths: {}".format(gp.theta.corr))
print("Covariance: {}".format(gp.theta.cov))
print("Nugget: {}".format(gp.theta.nugget))

# create 20 target points to predict

predict_points = ed.sample(n_preds)

means, variances, derivs = gp.predict(predict_points)

print_predictions(predict_points, means, variances)

###################################################################################

# Second Example: Specify Kernel using a string

print("Example 2: Product Matern Kernel")

# You may use a string matching the name of the Kernel type you wish to use

gp_matern = mogp_emulator.fit_GP_MAP(inputs, targets, kernel='ProductMat52', nugget=1.e-8)

# look at hyperparameters (correlation lengths, covariance, and nugget)

print("Correlation lengths: {}".format(gp_matern.theta.corr))
print("Covariance: {}".format(gp_matern.theta.cov))
print("Nugget: {}".format(gp_matern.theta.nugget))

# return type from predict method is an object with mean, unc, etc defined as attributes

means, variances, derivs = gp_matern.predict(predict_points)

print_predictions(predict_points, means, variances)

###################################################################################

# Third Example: Use a Kernel object

print("Example 3: Use a Kernel Object")

# The UniformSqExp object only has a single correlation length for all inputs

kern = UniformSqExp()

gp_uniform = mogp_emulator.GaussianProcess(inputs, targets, kernel=kern)

# fit hyperparameters

gp_uniform = mogp_emulator.fit_GP_MAP(gp_uniform)

# Note that only a single correlation length

print("Correlation length: {}".format(gp_uniform.theta.corr))
print("Covariance: {}".format(gp_uniform.theta.cov))
print("Nugget: {}".format(gp_uniform.theta.nugget))

# gp can be called directly if only the means are desired

means, variances, derivs = gp_uniform.predict(predict_points)

print_predictions(predict_points, means, variances)