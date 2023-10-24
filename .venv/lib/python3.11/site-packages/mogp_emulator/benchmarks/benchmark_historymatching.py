"""
Simple demos and sanity checks for the HistoryMatching class.

Provided methods:
    get_y_simulated_1D:
        A toy model that acts as the simulator output for constructing GPEs for
        1D data.

    get_y_simulated_2D:
        A toy model that acts as the simulator output for constructing GPEs for
        2D data.

    demo_1D:
        Follows the example of
        http://www.int.washington.edu/talks/WorkShops/int_16_2a/People/Vernon_I/Vernon2.pdf
        in setting up a simple test model, constructing a gaussian process to
        emulate it, and ruling out expectations of the GPE based on an
        historical observation.

    demo_2D:
        As demo_1D, however the toy model is expanded to take two inputs rather
        then 1 extending it to a second dimension. Exists primarily to confirm
        that the HistoryMatching class is functional with higher-dimensional
        parameter spaces.

"""

from mogp_emulator.HistoryMatching import HistoryMatching
from mogp_emulator.GaussianProcess import GaussianProcess
from mogp_emulator.fitting import fit_GP_MAP
try:
    from matplotlib import pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False
import numpy as np


def get_y_simulated_1D(x):
    n_points = len(x)
    f = np.zeros(n_points)
    for i in range(n_points):
        f[i] = np.sin(2.*np.pi*x[i] / 50.)
    return f

def demo_1D():
    # Create a gaussian process
    x_training = np.array([
        [0.],
        [10.],
        [20.],
        [30.],
        [43.],
        [50.]
    ])

    y_training = get_y_simulated_1D(x_training)

    gp = GaussianProcess(x_training, y_training)
    np.random.seed(47)
    gp = fit_GP_MAP(gp)

    # Define observation
    obs = [-0.8, 0.0004]

    # Coords to predict
    n_rand = 2000
    x_predict_min = -3
    x_predict_max = 53
    x_predict = np.random.rand(n_rand)
    x_predict = np.sort(x_predict, axis=0)
    x_predict *= (x_predict_max - x_predict_min)
    x_predict += x_predict_min
    x_predict = x_predict[:,None]

    coords = x_predict

    # Compute GPE expectations
    expectations = gp.predict(coords)

    # Compute Implausbility
    hm = HistoryMatching(obs=obs, expectations=expectations)
    I = hm.get_implausibility()
    NROY = hm.get_NROY()
    RO = hm.get_RO()

    print("Fraction of points ruled out {:6}".format(str(float(len(RO))/float(n_rand))))

    # Plotting

    if makeplots:
        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.subplots_adjust(hspace=0)
        x_hist_plot = [min(x_predict)[0], max(x_predict)[0]]
        y_hist_plot = [obs[0], obs[0]]
        y_hist_err = 3*np.sqrt(obs[1])
        y_hist_up = [val + y_hist_err for val in y_hist_plot]
        y_hist_dn = [val - y_hist_err for val in y_hist_plot]

        axs[0].plot(                # Horizontal line at value of y_obs
            x_hist_plot,
            y_hist_plot,
            color = 'black',
            label = 'observation'
        )

        axs[0].fill_between(        # Error bounds on y_obs
            x_hist_plot,
            y_hist_dn,
            y_hist_up,
            color='black',
            alpha=0.25
        )

        axs[0].plot(                # Simulator output
            coords,
            get_y_simulated_1D(coords),
            color = 'black',
            label = 'simulator'
        )

        axs[0].scatter(             # Training Data
            x_training,
            y_training,
            marker = '.',
            color  = 'black',
            label  = 'Training Data',
            s      = 100
        )

        axs[0].plot(                # GPE expectation
            coords,
            expectations[0],
            color = 'red',
            label = 'GPE'
        )

        axs[0].fill_between(        # GPE uncertainty
            coords[:,0],
            expectations[0] - 3*np.sqrt(expectations[1]),
            expectations[0] + 3*np.sqrt(expectations[1]),
            color = 'red',
            alpha = 0.5
        )

        axs[1].scatter(             # Implausibility
            coords,
            I,
            marker='.',
            color='black'
        )

        axs[1].scatter(
            coords[NROY],
            I[NROY],
            marker='.',
            color='green'
        )

        axs[1].plot(                # implausibility Threshold
            x_hist_plot,
            [3,3],
            color = 'green',
            label = 'implausibility threshold'
        )

        axs[0].set(
            ylabel='Model Output f(x)'
        )
        axs[1].set(
            xlabel='Imput Parameter x',
            ylabel='Implausibility I(x)',
            ylim=(-1, 21)
        )


        plt.savefig('histmatch_1D.png', bbox_inches = 'tight')

def get_y_simulated_2D(x):
    n_points = len(x[:,0])
    f = np.zeros(n_points)
    for i in range(n_points):
        f[i] = np.sin(x[i,0] ** (2.0) + x[i,1] ** (2.0))
    return f

def demo_2D():
    # Create a Gaussian Process
    x_training = np.array([
        [0., 0.],
        [1.5, 1.5],
        [3., 3.],
        [0., 1.5],
        [1.5, 0.],
        [0., 3.],
        [3., 0.],
        [3., 1.5],
        [1.5, 3.]
    ])

    y_training = get_y_simulated_2D(x_training)

    gp = GaussianProcess(x_training, y_training)
    np.random.seed(47)
    gp = fit_GP_MAP(gp)

    # Define observation
    obs = [0.1, 0.0004]

    # Coords to predict
    n_rand = 2000
    a_predict_min = 0
    a_predict_max = np.pi
    b_predict_min = a_predict_min
    b_predict_max = a_predict_max
    a_predict = np.random.rand(n_rand) * (a_predict_max - a_predict_min) + a_predict_min
    b_predict = np.random.rand(n_rand) * (b_predict_max - b_predict_min) + b_predict_min
    x_predict = np.concatenate((a_predict[:,None], b_predict[:,None]), axis=1)
    x_predict = np.concatenate((x_predict, x_training), axis=0)

    coords = x_predict

    # Compute GPE expectations
    expectations = gp.predict(coords)

    # Compute Implausbility
    hm = HistoryMatching(obs=obs, expectations=expectations)
    I = hm.get_implausibility()
    NROY = hm.get_NROY()
    RO = hm.get_RO()

    print("Fraction of points ruled out {:6}".format(str(float(len(RO))/float(n_rand))))

    # Plotting
    if makeplots:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        Axes3D.scatter(     # Training Data
            ax,
            x_training[:,0],
            x_training[:,1],
            y_training,
            color='black',
            marker = '.',
            s = 100
        )

        #Axes3D.scatter(      # GPE prediction
        #    ax,
        #    coords[:,0],
        #    coords[:,1],
        #    expectations[0],
        #    color='red',
        #    marker='.',
        #    s=2
        #)

        Axes3D.scatter(     # GPE prediction uncertainty
            ax,
            coords[:,0][RO],
            coords[:,1][RO],
            expectations[0][RO] + 3*np.sqrt(expectations[1][RO]),
            color='red',
            marker='.',
            s=1
        )
        Axes3D.scatter(
            ax,
            coords[:,0][RO],
            coords[:,1][RO],
            expectations[0][RO] - 3*np.sqrt(expectations[1][RO]),
            color='red',
            marker='.',
            s=1
        )
        Axes3D.scatter(
            ax,
            coords[:,0][NROY],
            coords[:,1][NROY],
            expectations[0][NROY] + 3*np.sqrt(expectations[1][NROY]),
            color='green',
            marker='.',
            s=1
        )
        Axes3D.scatter(
            ax,
            coords[:,0][NROY],
            coords[:,1][NROY],
            expectations[0][NROY] - 3*np.sqrt(expectations[1][NROY]),
            color='green',
            marker='.',
            s=1
        )

        Axes3D.set(
            ax,
            xlabel = 'Input parameter a',
            ylabel = 'Input parameter b',
            zlabel = 'Model output f(a, b)'
        )



        plt.savefig('histmatch_2D.png', bbox_inches = "tight")


if __name__ == '__main__':
    demo_1D()
    demo_2D()
