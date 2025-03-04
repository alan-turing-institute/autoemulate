import numpy as np
from autoemulate.compare import AutoEmulate
from autoemulate.experimental_design import LatinHypercube
from autoemulate.simulations.projectile import simulate_projectile
from autoemulate.emulators import GaussianProcessMT
from dataclasses import dataclass, asdict

class Simulator:

    def __init__(
        self, *, 
        in_dim: int, 
        out_dim: int, 
        lower_bounds: int, 
        upper_bounds: int
    ):
        '''
        Parameters
        ----------
        in_dim : int
            Dimension of the input space.
        out_dim : int
            Dimension of the output space.
        lower_bounds : list[float]
            Lower bounds of the input space.
        upper_bounds : list[float]
            Upper bounds of the input space.
        '''
        assert len(lower_bounds) == len(upper_bounds) == in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''
        Parameters
        ----------
        x : np.ndarray of shape (n_samples, in_dim)
            Input to the simulator.
        '''
        raise NotImplementedError
    
    def sample_out(self, n: int) -> np.ndarray:
        '''
        Samples input-output pairs (the graph) of the simulator uniformly over the input space.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        '''
        x = self.sample_in(n)
        y = self.__call__(x)
        return x, y
    
    def sample_in(self, n: int):
        '''
        Samples input points uniformly over the input space.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        '''
        return LatinHypercube([self.lower_bounds, self.upper_bounds]).sample(n)
    
class Projectile(Simulator):

    def __call__(self, x):
        # Should provide some multiprocessing here
        return np.array([simulate_projectile(xi) for xi in x]).reshape(-1, 1)

class Emulator:

    def predict(self, x: np.ndarray) -> np.ndarray:
        '''
        Parameters
        ----------
        x : np.ndarray of shape (n_samples, in_dim)
            Input to the emulator.

        Returns
        -------
        mu : np.ndarray of shape (n_samples, out_dim)
            Output of the emulator.
        sigma : None | np.ndarray
            None -> No uncertainty.
            (n_samples,) -> isotropic variance of the output of the emulator.
            (n_samples, out_dim) -> diagonal of the covariance matrix of the output of the emulator.
            (n_samples, out_dim, out_dim) -> covariance matrix of the output of the emulator.
        '''
        raise NotImplementedError
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        '''
        Fits the emulator to the input-output pairs.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, in_dim)
            Input to the emulator.
        y : np.ndarray of shape (n_samples, out_dim)
            Output of the emulator.
        '''
        raise NotImplementedError
    
class GaussianProcessEmulator(Emulator):

    def __init__(self):
        self.gp = GaussianProcessMT()

    def predict(self, x):
        return self.gp.predict(x, return_std=True)
    
    def fit(self, x, y):
        self.gp.fit(x, y)

class Experimenter:

    def __init__(
        self, *,
        simulator: Simulator,
        emulator: Emulator,
        n_samples: int
    ):
        '''
        Parameters
        ----------
        simulator : Simulator
            Simulator to emulate.
        emulator : Emulator
            Emulator to fit to the simulator.
        n_samples : int
            Number of posterior samples to draw.
        '''
        assert isinstance(simulator, Simulator)
        assert isinstance(emulator, Emulator)
        assert n_samples > 0
        self.simulator = simulator
        self.emulator = emulator
        self.n_samples = n_samples

    def update(self):

        # Predict
        obs_xs = self.simulator.sample_in(self.n_samples)
        pred_mean, pred_std = self.emulator.predict(obs_xs)

        # Query and update data
        query_x = obs_xs[np.argmax(pred_std), :]
