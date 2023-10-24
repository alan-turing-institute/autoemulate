'''This benchmark performs convergence tests using the pivoted
Cholesky routines applied to the 2D Branin function. Details of the 2D
Branin function can be found at
https://www.sfu.ca/~ssurjano/branin.html. The code samples the Branin
function using an increasing number of points, with a duplicate point
added to make the matrix singular. When pivoting is not used, the
algorithm is stabilized by adding a nugget term to the diagonal of the
covariance matrix.  This degrades the performance of the emulator
globally, despite the fact that the problem arises from a local
problem in fitting the emulator. Pivoting ignores points that are too
close to one another, ensuring that there is no loss of performance as
the number of points increases.

Note that this benchmark only covers relatively small designs. Tests
have revealed that there are some stability issues when applying
pivoting to larger numbers of inputs -- this appears to be due to the
minimization algorithm, perhaps due to the fact that pivoting computes
the inverse of a slightly different matrix which may influence the
fitting algorithm performance. Care should thus be taken to examine
the resulting performance when applying pivoting in practice. Future
versions may implement other approaches to ensure that pivoting gives
stable performance on a wide variety of input data.

'''

import numpy as np
from mogp_emulator import GaussianProcess, fit_GP_MAP
from mogp_emulator import MonteCarloDesign, LatinHypercubeDesign
from scipy.stats import uniform
try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

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


f = branin_2d
n_dim = 2
design_space = [uniform(loc = -5., scale = 15.).ppf, uniform(loc = 0., scale = 15.).ppf]
simulations = [5, 10, 15, 20, 25, 30]

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
    "Generate n_simulations input data and add a duplicate point to make matrix singular"
    
    inputs = generate_input_data(n_simulations, method = "lhd")
    targets = generate_target_data(inputs)
    
    inputs_new = np.zeros((inputs.shape[0] + 1, inputs.shape[1]))
    targets_new = np.zeros(targets.shape[0] + 1)
    
    inputs_new[:-1, :] = np.copy(inputs)
    targets_new[:-1] = np.copy(targets)
    
    inputs_new[-1,:] = np.copy(inputs[0,:])
    targets_new[-1] = np.copy(targets[0])
    
    return inputs_new, targets_new
    
def generate_test_data(n_testing):
    "Generate n_testing points for testing the accuracy of an emulator"
    
    testing = generate_input_data(n_testing, method = "random")
    test_targets = generate_target_data(testing)
    
    return testing, test_targets
    
def run_model(n_simulations, n_testing):
    "Generate training data, fit emulator, and test model accuracy on random points, returning RMSE"
    
    print('fitting GPs')
    
    # run LHD model
    inputs, targets = generate_training_data(n_simulations)
    
    gp = GaussianProcess(inputs, targets, nugget="adaptive")
    gp = fit_GP_MAP(gp)
    
    print("fitting pivoted GP")
    
    gp_pivot = GaussianProcess(inputs, targets, nugget="pivot")
    gp_pivot = fit_GP_MAP(gp_pivot)
    
    print("making predictions")

    testing, test_targets = generate_test_data(n_testing)
    
    norm_const = np.max(test_targets)-np.min(test_targets)
    
    test_vals_pivot, unc_pivot, deriv = gp_pivot.predict(testing, deriv = False, unc = True)
    test_vals, unc, deriv = gp.predict(testing, deriv = False, unc = True)
    
    return (np.sqrt(np.sum((test_vals - test_targets)**2)/float(n_testing))/norm_const,
            np.sqrt(np.sum(unc**2)/float(n_testing))/norm_const**2,
            np.sqrt(np.sum((test_vals_pivot - test_targets)**2)/float(n_testing))/norm_const,
            np.sqrt(np.sum(unc_pivot**2)/float(n_testing))/norm_const**2)

def plot_model_errors(simulation_list, error, unc, error_pivot, unc_pivot, n_testing):
    "Makes plot showing accuracy of emulator as a function of n_simulations"
    
    plt.figure(figsize=(4,3))
    plt.semilogy(simulation_list, error_pivot,'-o', label = 'Pivot')
    plt.semilogy(simulation_list, error,'-x', label = 'Nugget')
    plt.xlabel('Number of design points')
    plt.ylabel('Average prediction RMSE')
    plt.legend()
    plt.title('Error for '+str(n_testing)+' predictions')
    plt.savefig('pivot_error.png',bbox_inches='tight')
    
    plt.figure(figsize=(4,3))
    plt.semilogy(simulation_list, unc_pivot,'-o', label = "Pivot")
    plt.semilogy(simulation_list, unc,'-x', label = 'Nugget')
    plt.xlabel('Number of design points')
    plt.ylabel('Average prediction variance')
    plt.legend()
    plt.title('Uncertainty for '+str(n_testing)+' predictions')
    plt.savefig('pivot_unc.png',bbox_inches='tight')
    
def run_all_models(n_testing, simulation_list, n_iter = 10):
    "Runs all models, printing out results and optionally making plots"
    
    n_simtrials = len(simulation_list)
    
    errors = np.zeros((n_simtrials, n_iter))
    uncs = np.zeros((n_simtrials, n_iter))
    errors_pivot = np.zeros((n_simtrials, n_iter))
    uncs_pivot = np.zeros((n_simtrials, n_iter))
    
    for iteration in range(n_iter):
        for sim_index in range(n_simtrials):
            print(sim_index, iteration)
            (errors[sim_index, iteration],
             uncs[sim_index, iteration],
             errors_pivot[sim_index, iteration],
             uncs_pivot[sim_index, iteration]) = run_model(simulation_list[sim_index], n_testing)
    
    error = np.mean(errors, axis = -1)
    unc = np.mean(uncs, axis = -1)
    error_pivot = np.mean(errors_pivot, axis = -1)
    unc_pivot = np.mean(uncs_pivot, axis = -1)
    
    print("\n")
    print("Convergence test results:")
    print("Num. design points  RMSE Nugget         RMSE Pivot")
    for sim, err, pivot_err in zip(simulation_list, error, error_pivot):
        print('{:19} {:19} {:19}'.format(str(sim), str(err), str(pivot_err)))
        
    print("\n")
    print("Num. design points  Variance Nugget     Variance Pivot")
    for sim, un, pivot_un in zip(simulation_list, unc, unc_pivot):
        print('{:19} {:19} {:19}'.format(str(sim), str(un), str(pivot_un)))
    
    if makeplots:
        plot_model_errors(simulation_list, error, unc, error_pivot, unc_pivot, n_testing)
    
if __name__ == '__main__':
    run_all_models(100, [int(x) for x in simulations], n_iter = 10)
