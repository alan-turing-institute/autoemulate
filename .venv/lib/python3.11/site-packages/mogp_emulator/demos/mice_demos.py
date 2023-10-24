import mogp_emulator
import numpy as np
from projectile import simulator, print_results

# simple MICE examples using the projectile demo

# Base design -- requires a list of parameter bounds if you would like to use
# uniform distributions. If you want to use different distributions, you
# can use any of the standard distributions available in scipy to create
# the appropriate ppf function (the inverse of the cumulative distribution).
# Internally, the code creates the design on the unit hypercube and then uses
# the distribution to map from [0,1] to the real parameter space.

lhd = mogp_emulator.LatinHypercubeDesign([(-5., 1.), (0., 1000.)])

###################################################################################

# first example -- run entire design internally within the MICE class.

# first argument is base design (required), second is simulator function (optional,
# but required if you want the code to run the simualtions internally)

# Other optional arguments include:
# n_samples (number of sequential design steps, optional, default is not specified
# meaning that you will specify when running the sequential design)
# n_init (size of initial design, default 10)
# n_cand (number of candidate points, default is 50)
# nugget (nugget parameter for design GP, default is to set adaptively)
# nugget_s (nugget parameter for candidate GP, default is 1.)

n_init = 5
n_samples = 20
n_cand = 100

md = mogp_emulator.MICEDesign(lhd, simulator, n_samples=n_samples, n_init=n_init, n_cand=n_cand)

md.run_sequential_design()

# get design and outputs

inputs = md.get_inputs()
targets = md.get_targets()

print("Example 1:")
print("Design inputs:\n", inputs)
print("Design targets:\n", targets)
print()

###################################################################################

# second example: run design manually

md2 = mogp_emulator.MICEDesign(lhd, n_init=n_init, n_cand=n_cand)

init_design = md2.generate_initial_design()

print("Example 2:")
print("Initial design:\n", init_design)

# run initial points manually

init_targets = np.array([simulator(s) for s in init_design])

# set initial targets

md2.set_initial_targets(init_targets)

# run 20 sequential design steps

for d in range(n_samples):
    next_point = md2.get_next_point()
    next_target = simulator(next_point)
    md2.set_next_target(next_target)

# look at design and outputs

inputs = md2.get_inputs()
targets = md2.get_targets()

print("Final inputs:\n", inputs)
print("Final targets:\n", targets)

# look at final GP emulator and make some predictions to compare with lhd

lhd_design = lhd.sample(n_init + n_samples)

gp_lhd = mogp_emulator.fit_GP_MAP(lhd_design, np.array([simulator(p) for p in lhd_design]))

gp_mice = mogp_emulator.GaussianProcess(inputs, targets)

gp_mice = mogp_emulator.fit_GP_MAP(inputs, targets)

test_points = lhd.sample(10)

print("LHD:")
print_results(test_points, gp_lhd(test_points))
print()
print("MICE:")
print_results(test_points, gp_mice(test_points))