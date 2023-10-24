import mogp_emulator
import numpy as np

# simple History Matching example

# simulator function -- needs to take a single input and output a single number

def f(x):
    return np.exp(-np.sum((x-2.)**2, axis = -1)/2.)

# Experimental design -- requires a list of parameter bounds if you would like to use
# uniform distributions. If you want to use different distributions, you
# can use any of the standard distributions available in scipy to create
# the appropriate ppf function (the inverse of the cumulative distribution).
# Internally, the code creates the design on the unit hypercube and then uses
# the distribution to map from [0,1] to the real parameter space.

ed = mogp_emulator.LatinHypercubeDesign([(0., 5.), (0., 5.)])

# sample space, use many samples to ensure we get a good emulator

inputs = ed.sample(50)

# run simulation

targets = np.array([f(p) for p in inputs])

# Example observational data is a single number plus an uncertainty.
# In this case we use a number close to 1, which should have a corresponding
# input close to (2,2) after performing history matching

###################################################################################

# First step -- fit GP using MLE and Squared Exponential Kernel

gp = mogp_emulator.GaussianProcess(inputs, targets)

gp = mogp_emulator.fit_GP_MAP(gp)

###################################################################################

# First Example: Use HistoryMatching class to make the predictions

print("Example 1: Make predictions with HistoryMatching object")

# create HistoryMatching object, set threshold to be low to make printed output
# easier to read

threshold = 0.01
hm = mogp_emulator.HistoryMatching(threshold=threshold)

# For this example, we set the observations, GP, and the coordinates
# observations is either a single float (the value) or two floats (value and
# uncertainty as a variance)

obs = [1., 0.08]
hm.set_obs(obs)
hm.set_gp(gp)

# set coordinates of GP object where we will test if the points can plausbily
# explain the data here we use our existing experimental design, but sample
# 10000 points

coords = ed.sample(10000)
hm.set_coords(coords)

# calculate implausibility metric

implaus = hm.get_implausibility()

# print points that we have not ruled out yet:

for p, im in zip(coords[hm.get_NROY()], implaus[hm.get_NROY()]):
    print("Sample point: {}      Implausibility: {}".format(p, im))

###################################################################################

# Second Example: Pass external GP predictions and add model discrepancy

print("Example 2: External Predictions and Model Discrepancy")

# use gp to make predictions on 10000 new points externally

coords = ed.sample(10000)

expectations = gp.predict(coords)

# now create HistoryMatching object with these new parameters

hm_extern = mogp_emulator.HistoryMatching(obs=obs, expectations=expectations,
                                          threshold=threshold)

# calculate implausibility, adding a model discrepancy (as a variance)

implaus_extern = hm_extern.get_implausibility(0.1)

# print points that we have not ruled out yet:

for p, im in zip(coords[hm_extern.get_NROY()], implaus_extern[hm_extern.get_NROY()]):
    print("Sample point: {}      Implausibility: {}".format(p, im))
