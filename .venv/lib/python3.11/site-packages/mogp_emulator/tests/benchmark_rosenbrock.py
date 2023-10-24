'''
This benchmark performs convergence tests on a single emulator with variable numbers of input parameters.
The example is based on the Rosenbrock function (see https://www.sfu.ca/~ssurjano/rosen.html). This
function can be defined in an artibrary number of dimensions, so it provides a useful test for how
emulators based on increasing numbers of parameters perform as the size of the training data is varied.
As the number of training points increases, the prediction error and prediction variance should
decrease. However, this will depend on the number of dimensions in the function -- in general, the
size of the input space grows exponentially with the number of dimensions, while the samples
drawn here grow linearly with the number of dimensions. Thus, the higher dimensional emulators will
perform worse for the same number of samples per dimension.
'''

import numpy as np
from mogp_emulator import GaussianProcess, fit_GP_MAP
from mogp_emulator import MonteCarloDesign, LatinHypercubeDesign
try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

def rosenbrock(x):
    """
    Rosenbrock function in an arbitrary number of dimensions. Input x should be a 2D array of shape
    (n, D), where the first index represents the different places where the function is evaluated and
    the second index represents the dimensions of the function.
    see https://www.sfu.ca/~ssurjano/rosen.html for more information
    """
    x = np.array(x)
    if len(x.shape) == 1:
        x = np.reshape(x, (1, len(x)))
    assert len(x.shape) == 2
    return np.sum( 100. * ( x[:,1:] - x[:,:-1]**2 )**2 + ( x[:,:-1] - 1. )**2, axis = 1)


def generate_input_data(n_simulations, n_dimensions, method = "random"):
    "Generate n_simulations evaluation points the n_dimensions dimensional Rosenbrock function"

    n_simulations = int(n_simulations)
    n_dimensions = int(n_dimensions)
    assert(n_simulations > 0)
    assert(n_dimensions > 0)
    assert method == "random" or method == "lhd"

    if method == "random":
        ed = MonteCarloDesign(n_dimensions, (-5., 10.))
    elif method == "lhd":
        ed = LatinHypercubeDesign(n_dimensions, (-5., 10.))
    inputs = ed.sample(n_simulations)
    return inputs

def generate_training_data(n_simulations, n_dimensions):
    "Generate n_simulations input data for an n_dimensions dimensional Rosenbrock function"

    inputs = generate_input_data(n_simulations, n_dimensions, method = "lhd")
    targets = rosenbrock(inputs)

    return inputs, targets

def generate_test_data(n_testing, n_dimensions):
    "Generate n_testing points for testing the accuracy of an emulator"

    testing = generate_input_data(n_testing, n_dimensions, method = "random")
    test_targets = rosenbrock(testing)

    return testing, test_targets

def run_model(n_simulations, n_dimensions, n_testing):
    "Generate training data, fit emulator, and test model accuracy on random points, returning RMSE"

    inputs, targets = generate_training_data(n_simulations, n_dimensions)

    norm_const = np.mean(targets)

    gp = GaussianProcess(inputs, targets)
    gp = fit_GP_MAP(gp)

    testing, test_targets = generate_test_data(n_testing, n_dimensions)

    test_vals, unc, deriv = gp.predict(testing, deriv = False, unc = True)

    return (np.sqrt(np.sum((test_vals - test_targets)**2)/float(n_testing))/norm_const,
            np.sqrt(np.sum(unc**2)/float(n_testing))/norm_const**2)

def plot_model_errors(dimension_list, simulation_list, error, unc, n_testing):
    "Makes plot showing accuracy of emulator as a function of n_simulations"

    plt.figure(figsize=(4,3))
    for dim, err in zip(dimension_list, error):
        plt.semilogy(np.array(simulation_list), err, '-o', label = str(dim)+" dimensions")
    plt.legend()
    plt.xlabel('Number of simulations per dimension')
    plt.ylabel('Relative prediction RMSE')
    plt.title('Error for '+str(n_testing)+' predictions\nusing Rosenbrock function')
    plt.savefig('rosenbrock_error.png',bbox_inches='tight')

    plt.figure(figsize=(4,3))
    for dim, un in zip(dimension_list, unc):
        plt.semilogy(np.array(simulation_list), un, '-o', label = str(dim)+" dimensions")
    plt.legend()
    plt.xlabel('Number of simulations per dimension')
    plt.ylabel('Relative prediction variance')
    plt.title('Uncertainty for '+str(n_testing)+' predictions\nusing Rosenbrock function')
    plt.savefig('rosenbrock_unc.png',bbox_inches='tight')

def run_all_models(n_testing, dimension_list, simulation_list, n_iter = 10):
    "Runs all simulations, printing out results and optionally makes plots of results"

    n_simtrials = len(simulation_list)
    n_dimtrials = len(dimension_list)

    errors = np.zeros((n_dimtrials, n_simtrials, n_iter))
    uncs = np.zeros((n_dimtrials, n_simtrials, n_iter))

    for dim_index in range(n_dimtrials):
        for sim_index in range(n_simtrials):
            for iteration in range(n_iter):
                errors[dim_index, sim_index, iteration], uncs[dim_index, sim_index, iteration] = (
                                    run_model(simulation_list[sim_index]*dimension_list[dim_index], dimension_list[dim_index], n_testing))

    error = np.mean(errors, axis = -1)
    unc = np.mean(uncs, axis = -1)

    print("\n")
    print("Convergence test results:")
    print("Num. simulations/dim   Num. dimensions   Relative prediction RMSE")
    for dim, err_list in zip(dimension_list, error):
        for sim, err in zip(simulation_list, err_list):
            print('{:23}{:18}{}'.format(str(sim*dim), str(dim), str(err)))

    print("\n")
    print("Num. simulations/dim   Num. dimensions   Relative prediction variance")
    for dim, un_list in zip(dimension_list, unc):
        for sim, un in zip(simulation_list, un_list):
            print('{:23}{:18}{}'.format(str(sim*dim), str(dim), str(un)))

    if makeplots:
        plot_model_errors(dimension_list, simulation_list, error, unc, n_testing)

if __name__ == '__main__':
    run_all_models(100, [4, 6, 8], [i for i in range(2, 16)], 10)