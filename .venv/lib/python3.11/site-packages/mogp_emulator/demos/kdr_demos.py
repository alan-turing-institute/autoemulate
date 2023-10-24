import mogp_emulator
import numpy as np

# simple Dimension Reduction examples

# simulator function -- returns a single "important" dimension from
# at least 4 inputs

def f(x):
    return (x[0]-x[1]+2.*x[3])/3.

# Experimental design -- create a design with 5 input parameters
# all uniformly distributed over [0,1].

ed = mogp_emulator.LatinHypercubeDesign(5)

# sample space

inputs = ed.sample(100)

# run simulation

targets = np.array([f(p) for p in inputs])

###################################################################################

# First example -- dimension reduction given a specified number of dimensions
# (note that in real life, we do not know that the underlying simulation only
# has a single dimension)

print("Example 1: Basic Dimension Reduction")

# create DR object with a single reduced dimension (K = 1)

dr = mogp_emulator.gKDR(inputs, targets, K=1)

# use it to create GP

gp = mogp_emulator.fit_GP_MAP(dr(inputs), targets)

# create 5 target points to predict

predict_points = ed.sample(5)
predict_actual = np.array([f(p) for p in predict_points])

means = gp(dr(predict_points))

for pp, m, a in zip(predict_points, means, predict_actual):
    print("Target point: {} Predicted mean: {} Actual mean: {}".format(pp, m, a))

###################################################################################

# Second Example: Estimate dimensions from data

print("Example 2: Estimate the number of dimensions from the data")

# Use the tune_parameters method to use cross validation to create DR object
# Note this is more realistic than the above as it does not know the
# number of dimensions in advance

dr_tuned, loss = mogp_emulator.gKDR.tune_parameters(inputs, targets,
                                                    mogp_emulator.fit_GP_MAP,
                                                    cXs=[3.], cYs=[3.])

# Get number of inferred dimensions (usually gives 2)

print("Number of inferred dimensions is {}".format(dr_tuned.K))

# use object to create GP

gp_tuned = mogp_emulator.fit_GP_MAP(dr_tuned(inputs), targets)

# create 10 target points to predict

predict_points = ed.sample(5)
predict_actual = np.array([f(p) for p in predict_points])

means = gp_tuned(dr_tuned(predict_points))

for pp, m, a in zip(predict_points, means, predict_actual):
    print("Target point: {} Predicted mean: {} Actual mean: {}".format(pp, m, a))
