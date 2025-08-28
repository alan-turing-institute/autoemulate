import numpy as np
import torch
from scipy.integrate import solve_ivp

from autoemulate.core.types import NumpyLike, TensorLike
from autoemulate.simulations.base import Simulator


class DoublePendulum(Simulator):
    """
    Simulator of double pendulum motion.

    A double pendulum consists of two pendulums attached end to end. The motion
    is chaotic and highly sensitive to initial conditions. This simulator computes
    the time evolution of both pendulum angles and the total kinetic energy.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        log_level: str = "progress_bar",
        t_span: tuple[float, float] = (0.0, 10.0),
        n_time_points: int = 500,
        g: float = 9.81,
    ):
        """
        Initialize the double pendulum simulator.

        Parameters
        ----------
        parameters_range : dict[str, tuple[float, float]] | None
            Parameter ranges for m1, m2, l1, l2, theta1_0, theta2_0
        output_names : list[str] | None
            Names of output variables
        log_level : str
            Logging level
        t_span : tuple[float, float]
            Time span for simulation (start, end)
        n_time_points : int
            Number of time points to output
        g : float
            Gravitational acceleration
        """
        if parameters_range is None:
            parameters_range = {
                "m1": (0.06, 0.20),  # mass of first pendulum (kg)
                "m2": (0.06, 0.20),  # mass of second pendulum (kg)
                "l1": (0.5, 0.5),  # length of first pendulum (m)
                "l2": (0.5, 0.5),  # length of second pendulum (m)
                "theta1_0": (
                    -30 * np.pi / 180,
                    0,
                ),  # initial angle of first pendulum (rad)
                "theta2_0": (0, 0),  # initial angle of second pendulum (rad)
            }

        if output_names is None:
            # Create output names for time series
            theta1_names = [f"theta1_t{i}" for i in range(n_time_points)]
            theta2_names = [f"theta2_t{i}" for i in range(n_time_points)]
            ke_names = [f"kinetic_energy_t{i}" for i in range(n_time_points)]
            output_names = theta1_names + theta2_names + ke_names

        super().__init__(parameters_range, output_names, log_level)

        self.t_span = t_span
        self.n_time_points = n_time_points
        self.g = g
        self.time_points = np.linspace(t_span[0], t_span[1], n_time_points)

    def _forward(self, x: TensorLike) -> TensorLike | None:
        """
        Simulate the double pendulum motion and return time series.

        Parameters
        ----------
        x : TensorLike
            Input parameters [m1, m2, l1, l2, theta1_0, theta2_0]

        Returns
        -------
        TensorLike | None
            Time series data: [theta1_series, theta2_series, kinetic_energy_series]
        """
        assert x.shape[0] == 1, (
            f"Simulator._forward expects a single input, got {x.shape[0]}"
        )

        # Extract parameters
        params = x.cpu().numpy()[0]
        m1, m2, l1, l2, theta1_0, theta2_0 = params

        # Simulate the double pendulum
        try:
            theta1_series, theta2_series, ke_series = simulate_double_pendulum(
                m1, m2, l1, l2, theta1_0, theta2_0, self.time_points, self.g
            )

            # Concatenate all time series into a single output vector
            output = np.concatenate([theta1_series, theta2_series, ke_series])
            return torch.tensor(output).view(1, -1)

        except Exception as e:
            # If simulation fails, return None
            self.logger.warning("Double pendulum simulation failed: %s", e)
            return None


def double_pendulum_equations(y, m1, m2, l1, l2, g):
    """
    Solve double pendulum equations of motion.

    y = [theta1, z1, theta2, z2] where z1 = theta1_dot, z2 = theta2_dot.
    """
    theta1, z1, theta2, z2 = y

    # Calculate angle difference
    delta_theta = theta1 - theta2

    # Denominators for the equations
    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta_theta) * np.cos(delta_theta)
    den2 = (l2 / l1) * den1

    # Numerators for theta1_dot_dot
    num1 = (
        -m2 * l1 * z1**2 * np.sin(delta_theta) * np.cos(delta_theta)
        + m2 * g * np.sin(theta2) * np.cos(delta_theta)
        + m2 * l2 * z2**2 * np.sin(delta_theta)
        - (m1 + m2) * g * np.sin(theta1)
    )

    # Numerators for theta2_dot_dot
    num2 = (
        -m2 * l2 * z2**2 * np.sin(delta_theta) * np.cos(delta_theta)
        + (m1 + m2) * g * np.sin(theta1) * np.cos(delta_theta)
        - (m1 + m2) * l1 * z1**2 * np.sin(delta_theta)
        - (m1 + m2) * g * np.sin(theta2)
    )

    # Second derivatives
    theta1_dot_dot = num1 / den1
    theta2_dot_dot = num2 / den2

    return [z1, theta1_dot_dot, z2, theta2_dot_dot]


def compute_kinetic_energy(
    theta1: float,
    z1: float,
    theta2: float,
    z2: float,
    m1: float,
    m2: float,
    l1: float,
    l2: float,
) -> float:
    """Compute total kinetic energy of the double pendulum system."""
    # Velocities of first mass (only rotational)
    v1_sq = l1**2 * z1**2

    # Velocities of second mass (more complex due to compound motion)
    v2x = l1 * z1 * np.cos(theta1) + l2 * z2 * np.cos(theta2)
    v2y = l1 * z1 * np.sin(theta1) + l2 * z2 * np.sin(theta2)
    v2_sq = v2x**2 + v2y**2

    # Total kinetic energy
    return 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq


def simulate_double_pendulum(
    m1: float,
    m2: float,
    l1: float,
    l2: float,
    theta1_0: float,
    theta2_0: float,
    time_points: NumpyLike,
    g: float = 9.81,
) -> tuple[NumpyLike, NumpyLike, NumpyLike]:
    """Simulate double pendulum motion using your proven implementation."""
    # Initial conditions: [theta1, z1, theta2, z2]
    # Assume zero initial angular velocities
    y0 = [theta1_0, 0.0, theta2_0, 0.0]

    # Time span
    t_span = [time_points[0], time_points[-1]]

    # Solve using your exact implementation (FIXED: removed 't' parameter)
    solution = solve_ivp(
        fun=lambda y: double_pendulum_equations(y, m1, m2, l1, l2, g),
        t_span=t_span,
        y0=y0,
        t_eval=time_points,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")

    # Extract angle time series
    theta1_series = solution.y[0]
    theta2_series = solution.y[2]

    # Compute kinetic energy time series
    ke_series = np.zeros_like(time_points)
    for i, _t in enumerate(time_points):
        theta1, z1, theta2, z2 = solution.y[:, i]
        ke_series[i] = compute_kinetic_energy(theta1, z1, theta2, z2, m1, m2, l1, l2)

    return theta1_series, theta2_series, ke_series
