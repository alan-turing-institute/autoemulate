import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2
from numpy.fft import ifft2
from scipy.integrate import solve_ivp
from tqdm import tqdm

integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "RK45"
integrator_keywords["atol"] = 1e-12


def reaction_diffusion(t, uvt, K22, d1, d2, beta, n, N):
    """
    Define the reaction-diffusion PDE in the Fourier (kx, ky) space
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
    uvt_updated = np.squeeze(
        np.vstack(
            (-d1 * K22 * uvt_reshaped[:N] + utrhs, -d2 * K22 * uvt_reshaped[N:] + vtrhs)
        )
    )
    return uvt_updated


def simulate_reaction_diffusion(x, return_timeseries=False, n=32, L=20, T=10.0, dt=0.1):
    """ "
    Simulate the reaction-diffusion PDE for a given set of parameters

    Parameters
    ----------
    x : array-like
        The parameters of the reaction-diffusion model. The first element is the reaction coefficient (beta) and the second element is the diffusion coefficient (d).
    n : int
        Number of spatial points in each direction
    L : int
        Domain size in X and Y directions
    T : float
        Total time to simulate
    dt : float
        Time step size

    Returns
    -------
    u_sol : array-like
        The spatial solution of the reaction-diffusion PDE at the final time point
    v_sol : array-like
        The spatial solution of the reaction-diffusion PDE at the final time point
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

    # Solve the PDE in the Fourier space, where it rseduces to system of ODEs
    uvsol = solve_ivp(
        reaction_diffusion,
        (t[0], t[-1]),
        y0=uvt0,
        t_eval=t,
        args=(K22, d1, d2, beta, n, N),
        **integrator_keywords
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
    else:
        # Return the last snapshot
        u_sol = u[:, :, -1]
        v_sol = v[:, :, -1]
        return u_sol, v_sol
