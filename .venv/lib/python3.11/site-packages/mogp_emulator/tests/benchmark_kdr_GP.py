"""Dimension reduction benchmark (gKDR)

Plot L_1 model error against reduced dimension, for a one-hundred
dimensional problem.  The model is a Gaussian process (with
maximum-likelihood hyperparameters), with inputs determined by gKDR,
varying structural dimension and with a selection of scale parameters.
The inputs to the model are 100 points chosen uniformly at random in
100 dimensions, and the observations are a linear mapping of these
into one dimension.
"""

import os
from contextlib import redirect_stdout
import numpy as np
from mogp_emulator import fit_GP_MAP
from mogp_emulator import DimensionReduction
from mogp_emulator import gKDR
try:
    import matplotlib.pyplot as plt
    have_plt = True
except ImportError:
    print("matplotlib.pyplot import error: skipping plots")
    have_plt = False

dev_null = open(os.devnull, "w")

def fn(x):
    return x[0]


def make_plot(loss):
    """Produce a plot of loss as a function of structural dimension."""
    plt.xlabel("Structural dimension")
    plt.ylabel("Loss ($L_1$)")
    for ic in range(loss.shape[0]):
        assert(np.all(loss[ic,:,2:] == loss[ic,0,2:]))
        cX, cY = loss[ic,0,2:]
        plt.plot(loss[ic,:,0], loss[ic,:,1], 'x-', label = "$c_X = {}, c_Y = {}$".format(cX, cY))

    plt.ylim(bottom=0.0)
    plt.legend()
    # plt.show()
    plt.savefig("benchmark_kdr_GP_loss.pdf")


def run():
    """Run the benchmark"""
    N = 100
    D = 100
    np.random.seed(3)
    X = np.random.random((N, D))
    Y = np.apply_along_axis(fn, 1, X)

    def compute_loss(k):
        ## ignore occasional warning messages from the GP fitting
        with redirect_stdout(dev_null):
            loss = gKDR._compute_loss(X, Y, fit_GP_MAP,
                                      cross_validation_folds = 5,
                                      K = k, X_scale = cX, Y_scale = cY)

        print("loss(K={}, cX={}, cY={}) = {}".format(k, cX, cY, loss))
        return loss

    K = np.arange(1,101)
    scale_params = [(1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]

    loss = np.zeros((len(scale_params), len(K), 4))

    for ic in range(len(scale_params)):
        cX, cY = scale_params[ic]
        loss[ic,:,:] = np.array([[k, compute_loss(k), cX, cY] for k in K])

    np.save("benchmark_kdr_GP_loss", loss)

    if have_plt:
        make_plot(loss)


if __name__ == '__main__':
    run()
