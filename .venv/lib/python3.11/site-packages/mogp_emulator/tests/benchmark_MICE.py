'''
This benchmark performs convergence tests using the MICE experimental design applied to the
2D Branin function. Details of the 2D Branin function can be found at
https://www.sfu.ca/~ssurjano/branin.html. The code samples the Branin function using the
MICE experimental design algorithm with a varying number of target points, and then tests
the convergence of the resulting emulators. As the number of targe points increases, the
prediction error and prediction variance should decrease.

(Note however that eventually, the predictions worsen once the number of target points becomes
large enough that the points become too densely sampled. In this case, the points become
co-linear and the resulting covariance matrix is singular and cannot be inverted. To avoid
this problem, the code iteratively adds additional noise to the covariance function to
stabilize the inversion. However, this noise reduces the accuracy of the predictions. The
values chosen for this benchmark attempt to avoid this, but in some cases this still becomes
a problem due to the inherent smoothness of the squared exponential covariance function.)
'''

import numpy as np
from mogp_emulator import GaussianProcess, fit_GP_MAP
from mogp_emulator import MICEDesign, MonteCarloDesign, LatinHypercubeDesign
from scipy.stats import uniform
try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

problem = 'branin'

def branin_2d(x):
    "2D Branin function, see https://www.sfu.ca/~ssurjano/branin.html for more information"
    if np.array(x).shape == (2,):
        x1, x2 = x
    else:
        assert len(np.array(x).shape) == 2
        assert np.array(x).shape[1] == 2
        x1 = x[:,0]
        x2 = x[:,1]
    a, b, c, r, s, t = 1., 5.1/4./np.pi**2, 5./np.pi, 6., 10., 1./8./np.pi
    return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1. - t)*np.cos(x1) + s

def oscillatory_4d(x):
    "4d oscillatory function for testing"
    if np.array(x).shape == (4,):
        x1, x2, x3, x4 = x
    else:
        assert len(np.array(x).shape) == 2
        assert np.array(x).shape[1] == 4
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]
    a, b, c, d, w = 1.85, 2.51, 1.94, 2.7, 0.43
    return np.cos(a*x1 + b*x2 + c*x3 + d*x4 + 2.*np.pi*w)

if problem == 'branin':
    f = branin_2d
    n_dim = 2
    design_space = [uniform(loc = -5., scale = 15.).ppf, uniform(loc = 0., scale = 15.).ppf]
    simulations = [5, 10, 15, 20, 25, 30]
else:
    f = oscillatory_4d
    n_dim = 4
    design_space = 4
    simulations = [5, 10, 15, 20, 25, 30, 35]

def generate_input_data(n_simulations, method = "random"):
    "Generate random points x1 and x2 for evaluating the multivalued 2D Branin function"

    n_simulations = int(n_simulations)
    assert(n_simulations > 0)
    assert method == "random" or method == "lhd"

    if method == "random":
        ed = MonteCarloDesign(design_space)
    elif method == "lhd":
        ed = LatinHypercubeDesign(design_space)
    inputs = ed.sample(n_simulations)
    return inputs

def generate_target_data(inputs):
    "Generate target data for multivalued emulator benchmark"

    inputs = np.array(inputs)
    assert len(inputs.shape) == 2
    assert inputs.shape[1] == n_dim
    n_simulations = inputs.shape[0]

    targets = f(inputs)

    return targets

def generate_training_data(n_simulations):
    "Generate n_simulations input data and evaluate using n_emulators different parameter values"

    inputs = generate_input_data(n_simulations, method = "lhd")
    targets = generate_target_data(inputs)

    return inputs, targets

def generate_test_data(n_testing):
    "Generate n_testing points for testing the accuracy of an emulator"

    testing = generate_input_data(n_testing, method = "random")
    test_targets = generate_target_data(testing)

    return testing, test_targets

def run_model(n_simulations, n_testing):
    "Generate training data, fit emulator, and test model accuracy on random points, returning RMSE"

    # run MICE Model
    print('running MICE')
    ed = LatinHypercubeDesign(design_space)

    n_init = 5

    md = MICEDesign(ed, f, n_samples = n_simulations - n_init, n_init = n_init, n_cand = 100)

    md.run_sequential_design()

    inputs_mice = md.get_inputs()
    targets_mice = md.get_targets()

    print('fitting GPs')

    gp_mice = GaussianProcess(inputs_mice, np.squeeze(targets_mice))
    gp_mice = fit_GP_MAP(gp_mice)

    # run LHD model
    inputs, targets = generate_training_data(n_simulations)

    gp = GaussianProcess(inputs, targets)
    gp = fit_GP_MAP(gp)

    print("making predictions")

    testing, test_targets = generate_test_data(n_testing)

    norm_const = np.max(test_targets)-np.min(test_targets)

    test_vals_mice, unc_mice, deriv = gp_mice.predict(testing, deriv = False, unc = True)
    test_vals, unc, deriv = gp.predict(testing, deriv = False, unc = True)

    return (np.sqrt(np.sum((test_vals - test_targets)**2)/float(n_testing))/norm_const,
            np.sqrt(np.sum(unc**2)/float(n_testing))/norm_const**2,
            np.sqrt(np.sum((test_vals_mice - test_targets)**2)/float(n_testing))/norm_const,
            np.sqrt(np.sum(unc_mice**2)/float(n_testing))/norm_const**2)

def plot_model_errors(simulation_list, error, unc, error_mice, unc_mice, n_testing):
    "Makes plot showing accuracy of emulator as a function of n_simulations"

    plt.figure(figsize=(4,3))
    plt.semilogy(simulation_list, error_mice,'-o', label = 'MICE')
    plt.semilogy(simulation_list, error,'-x', label = 'LHD')
    plt.xlabel('Number of design points')
    plt.ylabel('Average prediction RMSE')
    plt.legend()
    plt.title('Error for '+str(n_testing)+' predictions')
    plt.savefig('mice_'+problem+'_error.png',bbox_inches='tight')

    plt.figure(figsize=(4,3))
    plt.semilogy(simulation_list, unc_mice,'-o', label = "MICE")
    plt.semilogy(simulation_list, unc,'-x', label = 'LHD')
    plt.xlabel('Number of design points')
    plt.ylabel('Average prediction variance')
    plt.legend()
    plt.title('Uncertainty for '+str(n_testing)+' predictions')
    plt.savefig('mice_'+problem+'_unc.png',bbox_inches='tight')

def run_all_models(n_testing, simulation_list, n_iter = 10):
    "Runs all models, printing out results and optionally making plots"

    n_simtrials = len(simulation_list)

    errors = np.zeros((n_simtrials, n_iter))
    uncs = np.zeros((n_simtrials, n_iter))
    errors_mice = np.zeros((n_simtrials, n_iter))
    uncs_mice = np.zeros((n_simtrials, n_iter))

    for iteration in range(n_iter):
        for sim_index in range(n_simtrials):
            print(sim_index, iteration)
            (errors[sim_index, iteration],
             uncs[sim_index, iteration],
             errors_mice[sim_index, iteration],
             uncs_mice[sim_index, iteration]) = run_model(simulation_list[sim_index], n_testing)

    error = np.mean(errors, axis = -1)
    unc = np.mean(uncs, axis = -1)
    error_mice = np.mean(errors_mice, axis = -1)
    unc_mice = np.mean(uncs_mice, axis = -1)

    print("\n")
    print("Convergence test results:")
    print("Num. design points  RMSE LHD            RMSE MICE")
    for sim, err, mice_err in zip(simulation_list, error, error_mice):
        print('{:19} {:19} {:19}'.format(str(sim), str(err), str(mice_err)))

    print("\n")
    print("Num. design points  Variance LHD        Variance MICE")
    for sim, un, mice_un in zip(simulation_list, unc, unc_mice):
        print('{:19} {:19} {:19}'.format(str(sim), str(un), str(mice_un)))

    if makeplots:
        plot_model_errors(simulation_list, error, unc, error_mice, unc_mice, n_testing)

if __name__ == '__main__':
    run_all_models(100, [int(x) for x in simulations], n_iter = 10)