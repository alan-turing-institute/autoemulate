import numpy as np
import scipy.sparse as sp
import torch
from scipy.fft import fft2, ifft2
from scipy.integrate import solve_ivp

from autoemulate.core.types import NumpyLike, TensorLike
from autoemulate.simulations.base import Simulator

integrator_keywords = {}
integrator_keywords["rtol"] = 1e-6
integrator_keywords["method"] = "RK45"
integrator_keywords["atol"] = 1e-8


class AdvectionDiffusion(Simulator):
    """Simulate the 2D vorticity equation (advection-diffusion)."""

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        n: int = 50,
        L: float = 10.0,
        T: float = 80.0,
        dt: float = 0.25,
    ):
        """
        Initialize the AdvectionDiffusion simulator.

        Parameters
        ----------
        parameters_range: dict[str, tuple[float, float]]
            Mapping of input parameter names to (min, max) ranges.
        output_names: list[str]
            List of output parameter names.
        log_level: str
            Logging level for the simulator.
        return_timeseries: bool
            Whether to return the full timeseries or just the final snapshot.
        n: int
            Number of spatial points per direction.
        L: float
            Domain size in X and Y directions.
        T: float
            Total simulation time.
        dt: float
            Time step size.
        """
        if parameters_range is None:
            parameters_range = {
                "nu": (0.0001, 0.01),  # viscosity
                "mu": (0.5, 2.0),  # advection strength
            }
        if output_names is None:
            output_names = ["solution"]
        super().__init__(parameters_range, output_names, log_level)
        self.return_timeseries = return_timeseries
        self.n = n
        self.L = L
        self.T = T
        self.dt = dt

    def _forward(self, x: TensorLike) -> TensorLike:
        assert x.shape[0] == 1, (
            f"Simulator._forward expects a single input, got {x.shape[0]}"
        )

        vorticity_sol = simulate_advection_diffusion(
            x.cpu().numpy()[0], self.return_timeseries, self.n, self.L, self.T, self.dt
        )

        return torch.tensor(vorticity_sol.ravel(), dtype=torch.float32).reshape(1, -1)

    def forward_samples_spatiotemporal(
        self, n: int, random_seed: int | None = None
    ) -> dict:
        """Reshape to spatiotemporal format and return data plus constants."""
        x = self.sample_inputs(n, random_seed)

        y, x = self.forward_batch(x)

        if self.return_timeseries:
            n_time = int(self.T / self.dt)
            y_reshaped = y.reshape(y.shape[0], n_time, self.n, self.n, 1)
        else:
            y_reshaped = y.reshape(y.shape[0], 1, self.n, self.n, 1)

        return {
            "data": y_reshaped,
            "constant_scalars": x,
            "constant_fields": None,
        }


def create_sparse_matrices(
    n: int, N: int
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Create sparse matrices A, Dx, and Dy for finite-difference operators."""
    e1 = np.ones(N)
    e2 = np.ones(N)
    e4 = np.zeros(N)

    for j in range(1, n + 1):
        e2[n * j - 1] = 0
        e4[n * j - 1] = 1

    e3 = np.zeros(N)
    e3[1:] = e2[:-1]
    e3[0] = e2[-1]

    e5 = np.zeros(N)
    e5[1:] = e4[:-1]
    e5[0] = e4[-1]

    # Create Laplacian matrix A
    diagonals = [e1, e1, e5, e2, -4 * e1, e3, e4, e1, e1]
    offsets = [-(N - n), -n, -n + 1, -1, 0, 1, n - 1, n, N - n]
    A = sp.diags(diagonals, offsets, shape=(N, N), format="csr").tolil()  # type: ignore[arg-type]
    A[0, 0] = 2

    # Create Dx matrix
    diagonals_x = [e1, -e1, e1, -e1]
    offsets_x = [-(N - n), -n, n, N - n]
    Dx = sp.diags(diagonals_x, offsets_x, shape=(N, N), format="csr").tolil()  # type: ignore[arg-type]

    # Create Dy matrix
    diagonals_y = [e5, -e2, e3, -e4]
    offsets_y = [-n + 1, -1, 1, n - 1]
    Dy = sp.diags(diagonals_y, offsets_y, shape=(N, N), format="csr").tolil()  # type: ignore[arg-type]

    return A.tocsr(), Dx.tocsr(), Dy.tocsr()


def advection_diffusion(
    _t: float,
    w2: NumpyLike,
    A: sp.csr_matrix,
    Dx: sp.csr_matrix,
    Dy: sp.csr_matrix,
    nu: float,
    dx: float,
    n: int,
    N: int,
    K3: NumpyLike,
    mu: float,
) -> NumpyLike:
    """
    Define the advection-diffusion RHS used by the ODE integrator.

    Parameters
    ----------
    _t: float
        Current time (unused).
    w2: NumpyLike
        Flattened vorticity field.
    A, Dx, Dy: sp.csr_matrix
        Sparse differential operators.
    nu: float
        Viscosity coefficient.
    dx: float
        Spatial step.
    n, N: int
        Grid sizes.
    K3: NumpyLike
        Inverse Laplacian in Fourier space.
    mu: float
        Advection strength.
    """
    w_2d = w2.reshape(n, n)

    # Compute stream function using FFT (Poisson solver)
    psi_2d = np.real(np.asarray(ifft2(-fft2(w_2d) * K3)))  # type: ignore[arg-type]
    psi2 = psi_2d.reshape(N)

    # Diffusion term + nonlinear advection terms
    return np.asarray(
        (nu / dx**2) * (A @ w2)
        - (0.25 / dx**2) * (Dx @ psi2) * (Dy @ w2) * mu
        + (0.25 / dx**2) * (Dy @ psi2) * (Dx @ w2) * mu
    )


def simulate_advection_diffusion(
    x: NumpyLike,
    return_timeseries: bool = False,
    n: int = 50,
    L: float = 10.0,
    T: float = 80.0,
    dt: float = 0.25,
) -> NumpyLike:
    """
    Simulate the 2D vorticity equation (advection-diffusion).

    Parameters
    ----------
    x: NumpyLike
        [nu, mu] parameters.
    return_timeseries: bool
        Whether to return full timeseries or only final snapshot.
    """
    nu, mu = x

    # Time vector
    tspan = np.arange(0, T, dt)
    n_time = len(tspan)

    # Spatial grid
    x_grid = np.linspace(-L / 2, L / 2, n)
    dx = float(x_grid[1] - x_grid[0])
    y_grid = x_grid
    N = n * n

    # Initial conditions - Gaussian vortex
    X, Y = np.meshgrid(x_grid, y_grid)
    w_initial = np.exp(-(X**2) - Y**2 / 20)

    # Create sparse matrices for finite differences
    A, Dx, Dy = create_sparse_matrices(n, N)

    # Wavenumber grid for FFT (Poisson solver)
    k = (2 * np.pi / L) * np.concatenate([np.arange(0, n // 2), np.arange(-n // 2, 0)])
    k[0] = 1e-6  # Avoid division by zero
    KX, KY = np.meshgrid(k, k)
    K3 = 1.0 / (KX**2 + KY**2)

    # Reshape initial condition
    w2_initial = w_initial.reshape(N)

    # Solve the ODE system
    sol = solve_ivp(
        lambda t, w2: advection_diffusion(t, w2, A, Dx, Dy, nu, dx, n, N, K3, mu),
        [0, T],
        w2_initial,
        t_eval=tspan,
        **integrator_keywords,
    )

    if return_timeseries:
        return sol.y.T.reshape(n_time, n, n)
    return sol.y[:, -1].reshape(n, n)
