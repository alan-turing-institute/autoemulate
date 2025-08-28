import numpy as np
import torch
from numpy.fft import fft2, ifft2
from scipy.integrate import solve_ivp

from autoemulate.core.types import NumpyLike, TensorLike
from autoemulate.simulations.base import Simulator

integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "RK45"
integrator_keywords["atol"] = 1e-12


class ReactionDiffusion(Simulator):
    """Simulate the reaction-diffusion PDE for a given set of parameters."""

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        n: int = 32,
        L: int = 20,
        T: float = 10.0,
        dt: float = 0.1,
    ):
        """
        Initialize the ReactionDiffusion simulator.

        Parameters
        ----------
        parameters_range: dict[str, tuple[float, float]]
            Dictionary mapping input parameter names to their (min, max) ranges.
        output_names: list[str]
            List of output parameters' names.
        log_level: str
            Logging level for the simulator. Can be one of:
            - "progress_bar": shows a progress bar during batch simulations
            - "debug": shows debug messages
            - "info": shows informational messages
            - "warning": shows warning messages
            - "error": shows error messages
            - "critical": shows critical messages
        return_timeseries: bool
            Whether to return the full timeseries or just the spatial solution at the
            final time step. Defaults to False.
        n: int
            Number of spatial points in each direction.
        L: int
            Domain size in X and Y directions.
        T: float
            Total time to simulate.
        dt: float
            Time step size.
        """
        if parameters_range is None:
            parameters_range = {"beta": (1.0, 2.0), "d": (0.05, 0.3)}
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
        u_sol, v_sol = simulate_reaction_diffusion(
            x.cpu().numpy()[0], self.return_timeseries, self.n, self.L, self.T, self.dt
        )

        # concatenate U and V arrays (flattened across time and space)
        concat_array = np.concatenate([u_sol.ravel(), v_sol.ravel()])

        # return tensor shape (1, 2*self.t*self.n*self.n)
        return torch.tensor(concat_array, dtype=torch.float32).reshape(1, -1)


def reaction_diffusion(
    t: float,  # noqa: ARG001
    uvt: NumpyLike,
    K22: NumpyLike,
    d1: float,
    d2: float,
    beta: float,
    n: int,
    N: int,
):
    """
    Define the reaction-diffusion PDE in the Fourier (kx, ky) space.

    Parameters
    ----------
    t: float
        The current time step (not used).
    uvt: NumpyLike
        Fourier transformed solution vector at current time step.
    K22: NumpyLike
        The squared magnitudes of the Fourier wavevectors (kx, ky).
    d1: float
        The diffusion coefficient for species 1.
    d2: float
        The diffusion coefficient for species 2.
    beta: float
        The reaction coefficient controlling reaction between the two species.
    n: int
        Number of spatial points in each direction.
    N: int
        Total number of spatial grid points (n*n).
    """
    ut = np.reshape(uvt[:N], (n, n))
    vt = np.reshape(uvt[N : 2 * N], (n, n))
    u = np.real(ifft2(ut))
    v = np.real(ifft2(vt))
    u3 = u**3
    v3 = v**3
    u2v = (u**2) * v
    uv2 = u * (v**2)
    utrhs = np.reshape((fft2(u - u3 - uv2 + beta * u2v + beta * v3)), (N, 1))
    vtrhs = np.reshape((fft2(v - u2v - v3 - beta * u3 - beta * uv2)), (N, 1))
    uvt_reshaped = np.reshape(uvt, (len(uvt), 1))
    return np.squeeze(
        np.vstack(
            (-d1 * K22 * uvt_reshaped[:N] + utrhs, -d2 * K22 * uvt_reshaped[N:] + vtrhs)
        )
    )


def simulate_reaction_diffusion(
    x: NumpyLike,
    return_timeseries: bool = False,
    n: int = 32,
    L: int = 20,
    T: float = 10.0,
    dt: float = 0.1,
) -> tuple[NumpyLike, NumpyLike]:
    """
    Simulate the reaction-diffusion PDE for a given set of parameters.

    Parameters
    ----------
    x: NumpyLike
        The parameters of the reaction-diffusion model. The first element is the
        reaction coefficient (beta) and the second element is the diffusion
        coefficient (d).
    return_timeseries: bool
        Whether to return the full timeseries or just the spatial solution at the final
        time step. Defaults to False.
    n: int
        Number of spatial points in each direction. Defaults to 32.
    L: int
        Domain size in X and Y directions. Defaults to 20.
    T: float
        Total time to simulate. Defaults to 10.0.
    dt: float
        Time step size. Defaults to 0.1.

    Returns
    -------
    tuple[NumpyLike, NumpyLike]
        [u_sol, v_sol], the spatial solution of the reaction-diffusion PDE, either as a
        timeseries or at the final time point of `return_timeseries` is False.
    """
    beta, d = x
    d1 = d2 = d

    t = np.linspace(0, T, int(T / dt))

    N = n * n
    x_uniform = np.linspace(-L / 2, L / 2, n + 1)
    x_grid = x_uniform[:n]
    y_grid = x_uniform[:n]
    n2 = int(n / 2)
    # Define Fourier wavevectors (kx, ky)
    kx = (2 * np.pi / L) * np.hstack(
        (np.linspace(0, n2 - 1, n2), np.linspace(-n2, -1, n2))
    )
    ky = kx
    # Get 2D meshes in (x, y) and (kx, ky)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K22 = np.reshape(K2, (N, 1))

    m = 1

    # define our solution vectors
    u = np.zeros((len(x_grid), len(y_grid), len(t)))
    v = np.zeros((len(x_grid), len(y_grid), len(t)))

    # Initial conditions
    u[:, :, 0] = np.tanh(np.sqrt(X_grid**2 + Y_grid**2)) * np.cos(
        m * np.angle(X_grid + 1j * Y_grid) - (np.sqrt(X_grid**2 + Y_grid**2))
    )
    v[:, :, 0] = np.tanh(np.sqrt(X_grid**2 + Y_grid**2)) * np.sin(
        m * np.angle(X_grid + 1j * Y_grid) - (np.sqrt(X_grid**2 + Y_grid**2))
    )

    # uvt is the solution vector in Fourier space, so below
    # we are initializing the 2D FFT of the initial condition, uvt0
    uvt0 = np.squeeze(
        np.hstack(
            (np.reshape(fft2(u[:, :, 0]), (1, N)), np.reshape(fft2(v[:, :, 0]), (1, N)))
        )
    )

    # Solve the PDE in the Fourier space, where it reduces to system of ODEs
    uvsol = solve_ivp(
        reaction_diffusion,
        (t[0], t[-1]),
        y0=uvt0,
        t_eval=t,
        args=(K22, d1, d2, beta, n, N),
        **integrator_keywords,
    )
    uvsol = uvsol.y

    # Reshape things and ifft back into (x, y, t) space from (kx, ky, t) space
    for j in range(len(t)):
        ut = np.reshape(uvsol[:N, j], (n, n))
        vt = np.reshape(uvsol[N:, j], (n, n))
        u[:, :, j] = np.real(ifft2(ut))
        v[:, :, j] = np.real(ifft2(vt))

    if return_timeseries:
        return u.transpose(2, 0, 1), v.transpose(2, 0, 1)
    # Return the last snapshot
    u_sol = u[:, :, -1]
    v_sol = v[:, :, -1]
    return u_sol, v_sol
