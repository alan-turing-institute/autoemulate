import numpy as np
import matplotlib.pyplot as plt

def simulate_greyscott(x=(0.60,0.62), return_last_snap=True, size=64, dt=1.0, steps=1000, seed_size=20):
    """
    Simulates the Gray-Scott reaction-diffusion system and returns the full time evolution of U and V.

    Parameters:
    -----------
    x : tuple, optional
        Feed and kill parameters of the Gray-Scott model (default = (0.06, 0.062)).

    size : int, optional
        Size of the square grid (default = 64).
        
    dt : float, optional
        Time step for integration (default = 1.0).
        
    steps : int, optional
        Number of simulation steps (default = 1000).
        
    seed_size : int, optional
        Size of the initial disturbance at the center of the grid (default = 20).


    Returns:
    --------
    U_all : np.ndarray
        Concentration of species U over time, shape = (steps, size, size).
        
    V_all : np.ndarray
        Concentration of species V over time, shape = (steps, size, size).

    Example:
    --------
    U, V = simulate_greyscott(size=256, F=0.06, k=0.062)
    """

    F, k = x

    # Fixed diffusion rates
    Du = 0.16
    Dv = 0.08
    
    # Fix the random seed for reproducibility
    np.random.seed(42)
    
    # Initialize concentrations
    U = np.ones((size, size), dtype=np.float32)
    V = np.zeros((size, size), dtype=np.float32)

    # Seed an initial pattern at the center
    r = seed_size
    U[size//2-r:size//2+r, size//2-r:size//2+r] = 0.50
    V[size//2-r:size//2+r, size//2-r:size//2+r] = 0.25
    
    # Add small, deterministic perturbation based on fixed seed
    U += 0.01 * np.sin(np.linspace(0, 2 * np.pi, size))[:, None]
    V += 0.01 * np.cos(np.linspace(0, 2 * np.pi, size))[None, :]

    # Allocate arrays to store the full time evolution
    U_all = np.zeros((steps, size, size), dtype=np.float32)
    V_all = np.zeros((steps, size, size), dtype=np.float32)

    def laplacian(Z):
        """Compute Laplacian using a 5-point stencil."""
        return (
            -4 * Z
            + np.roll(Z, (0, 1), (0, 1))
            + np.roll(Z, (0, -1), (0, 1))
            + np.roll(Z, (1, 0), (0, 1))
            + np.roll(Z, (-1, 0), (0, 1))
        )

    # Simulation loop
    for i in range(steps):
        Lu = laplacian(U)
        Lv = laplacian(V)

        reaction = U * V**2
        U += (Du * Lu - reaction + F * (1 - U)) * dt
        V += (Dv * Lv + reaction - (F + k) * V) * dt

        # Store current state
        U_all[i] = U
        V_all[i] = V
    
    if return_last_snap:
        return U_all[-1], V_all[-1]
    else:
        return U_all, V_all