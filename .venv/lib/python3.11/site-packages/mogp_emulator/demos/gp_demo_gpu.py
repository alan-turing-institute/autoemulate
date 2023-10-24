import numpy as np
import mogp_emulator
from projectile import simulator, print_results

# GP example using the projectile demo on a GPU

# To run this demo you must be on a machine with an Nvidia GPU, and with
# CUDA libraries available.  There are also dependencies on eigen and pybind11
# If you are working on a managed cluster, these may be available via commands
#
# module load cuda/11.2
# module load py-pybind11-2.2.4-gcc-5.4.0-tdtz6iq
# module load gcc/7
# module load eigen
#
# You should then be able to compile the cuda code at the same time as installing the mogp_emulator package, by doing (from the main mogp_emulator/ directory:
# pip install .
# (note that if you don't have write access to the global directory
# (e.g. if you are on a cluster such as CSD3), you should add the
# `--user` flag to this command)


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

# Basic example -- fit GP using MLE and Squared Exponential Kernel and predict

print("Example: Basic GP")

# create GP and then fit using MLE
# the only difference between this and the standard CPU implementation
# is to use the GaussianProcessGPU class rather than GaussianProcess.

gp = mogp_emulator.GaussianProcessGPU(inputs, targets)

gp = mogp_emulator.fit_GP_MAP(gp)

# create 20 target points to predict

predict_points = ed.sample(n_preds)

means, variances, derivs = gp.predict(predict_points)

print_results(predict_points, means)
