import numpy as np
import mogp_emulator
from mogp_emulator.demos.projectile import simulator, print_predictions

# additional GP examples using the projectile demo

# define some common variables

n_samples = 20
n_preds = 10

# Experimental design -- requires a list of parameter bounds if you would like to use
# uniform distributions. If you want to use different distributions, you
# can use any of the standard distributions available in scipy to create
# the appropriate ppf function (the inverse of the cumulative distribution).
# Internally, the code creates the design on the unit hypercube and then uses
# the distribution to map from [0,1] to the real parameter space.

ed = mogp_emulator.LatinHypercubeDesign([(-5., 1.), (0., 1000.)])

# sample space

inputs = ed.sample(n_samples)

# run simulation

targets = np.array([simulator(p) for p in inputs])

###################################################################################

# First example -- fit GP using MLE and Squared Exponential Kernel and predict

print("Example 1: Basic GP")

# create GP and then fit using MLE

gp = mogp_emulator.GaussianProcess(inputs, targets)

gp = mogp_emulator.fit_GP_MAP(gp)

# create 20 target points to predict

predict_points = ed.sample(n_preds)

means, variances, derivs = gp.predict(predict_points)

print_predictions(predict_points, means, variances)

###################################################################################

# Second Example: How to change the kernel, use a fixed nugget, and create directly using fitting function

print("Example 2: Matern Kernel")

# you can simply pass the args to GP to the fitting function

gp_matern = mogp_emulator.fit_GP_MAP(inputs, targets, kernel='Matern52', nugget=1.e-8)

# return type from predict method is an object with mean, unc, etc defined as attributes

pred_res = gp_matern.predict(predict_points)

print_predictions(predict_points, pred_res.mean, pred_res.unc)

###################################################################################

# Third Example: Specify a mean function and set priors to Fit Hyperparameters via MAP

print("Example 3: Mean Function and MAP fitting")

# This example uses a linear mean function and sets priors on the hyperparameters

# Linear mean has 3 hyperparameters (intercept and 2 slopes, one for each input)
# Kernel has 3 hyperparameters (2 correlation lengths, 1 covariance scale)
# Nugget is the final hyperparameter (7 in total)

# Use a normal prior on all mean function values (requires mean, std)
# Use a normal prior on correlation lengths (which are on a log scale, so becomes a lognormal
# distribution once raw values on log scale are converted to linear scale)
# Inverse Gamma distribution on covariance (favors large values)
# Gamma distribution on nugget (favors negative values)

priors = mogp_emulator.Priors.GPPriors(mean=mogp_emulator.Priors.MeanPriors(mean=np.zeros(3),
                                                                            cov=np.array([1., 1., 1.])),
                                       corr=[ mogp_emulator.Priors.LogNormalPrior(1., 1.),
                                              mogp_emulator.Priors.LogNormalPrior(1., 1.) ],
                                       cov=mogp_emulator.Priors.InvGammaPrior(1., 1.),
                                       nugget=mogp_emulator.Priors.GammaPrior(1., 1.))

# create GP, passing list of priors and a string representing the mean function
# tell it to estimate the nugget as well

gp_map = mogp_emulator.GaussianProcess(inputs, targets, mean="x[0]+x[1]", priors=priors, nugget="fit")

# fit hyperparameters

gp_map = mogp_emulator.fit_GP_MAP(gp_map)

# gp can be called directly if only the means are desired

pred_means = gp_map(predict_points)

print_predictions(predict_points, pred_means, [""]*n_preds)