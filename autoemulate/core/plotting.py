import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.core.getipython import get_ipython
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from autoemulate.core.types import NumpyLike, TensorLike
from autoemulate.emulators.base import Emulator
from autoemulate.simulations.base import Simulator


def display_figure(fig: Figure):
    """
    Display a matplotlib figure.

    Display a matplotlib figure appropriately based on the environment
    (Jupyter notebook or terminal).

    Parameters
    ----------
    fig: Figure
        The object to display.

    Returns
    -------
    Figure
        The input figure object.
    """
    # Are we in Jupyter?
    try:
        is_jupyter = get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        is_jupyter = False

    if is_jupyter:
        # we don't show otherwise it will double plot
        plt.close(fig)
        return fig
    # in terminal, show the plot
    plt.close(fig)
    plt.show()
    return fig


def plot_xy(
    x: NumpyLike,
    y: NumpyLike,
    y_pred: NumpyLike,
    y_variance: NumpyLike | None = None,
    ax: Axes | None = None,
    title: str = "xy",
    input_index: int | None = None,
    output_index: int | None = None,
    r2_score: float | None = None,
):
    """
    Plot observed and predicted values vs. features.

    Parameters
    ----------
    x: NumpyLike
        An array of inputs.
    y: NumpyLike
        An array of outputs.
    y_pred: NumpyLike
        An array of predictions.
    y_variance: NumpyLike | None
        An optional array of predictive variances.
    ax: Axes | None
        An optional matplotlib Axes object to plot on.
    title: str
        An optional title for the plot.
    input_index: int | None
        An optional index of the input dimension to plot.
    output_index: int | None
        An optional index of the output dimension to plot.
    r2_score: float | None
        An option r2 score to include in the plot legend.
    """
    # Sort the data
    sort_idx = np.argsort(x).flatten()
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    y_std = None
    if y_variance is not None:
        y_variance_sorted = y_variance[sort_idx]
        y_std = np.sqrt(y_variance_sorted)

    org_points_color = "Goldenrod"
    pred_points_color = "#6A5ACD"
    pred_line_color = "#6A5ACD"
    ci_color = "lightblue"

    assert ax is not None, "ax must be provided"
    if y_std is not None:
        ax.fill_between(
            x_sorted,
            y_pred_sorted - 2 * y_std,
            y_pred_sorted + 2 * y_std,
            color=ci_color,
            alpha=0.25,
            label="95% Confidence Interval",
        )
    ax.plot(
        x_sorted,
        y_pred_sorted,
        color=pred_line_color,
        label="pred.",
        alpha=0.8,
        linewidth=1,
    )  # , linestyle='--'
    ax.scatter(
        x_sorted,
        y_sorted,
        color=org_points_color,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        label="data",
    )
    ax.scatter(
        x_sorted,
        y_pred_sorted,
        color=pred_points_color,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7,
        label="pred.",
    )

    ax.set_xlabel(f"$x_{input_index}$", fontsize=13)
    ax.set_ylabel(f"$y_{output_index}$", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    # Get the handles and labels for the scatter plots
    handles, _ = ax.get_legend_handles_labels()

    # Add legend and get its bounding box
    lbl = "pred." if y_variance is None else "pred. (±2σ)"  # noqa: RUF001
    legend = ax.legend(
        handles[-2:],
        ["data", lbl],
        loc="best",
        handletextpad=0,
        columnspacing=0,
        ncol=2,
    )

    # Place R² just below the legend
    if legend:
        # Get the bounding box of the legend in axes coordinates
        bbox = legend.get_window_extent(ax.figure.canvas.get_renderer())  # pyright: ignore[reportAttributeAccessIssue]
        inv = ax.transAxes.inverted()
        bbox_axes = bbox.transformed(inv)
        # Place the text just below the legend
        text_x = bbox_axes.x0
        text_y = bbox_axes.y0 - 0.04  # small offset below legend
        ax.text(
            text_x,
            text_y,
            f"R\u00b2 = {r2_score:.6f}",
            transform=ax.transAxes,
            verticalalignment="top",
        )
    else:
        # fallback: place in lower left
        ax.text(
            0.05,
            0.05,
            f"R\u00b2 = {r2_score:.6f}",
            transform=ax.transAxes,
            verticalalignment="bottom",
        )


def calculate_subplot_layout(n_plots, n_cols=3):
    """
    Calculate optimal number of rows and columns for subplots.

    Parameters
    ----------
    n_plots: int
        Number of plots to display.
    n_cols: int
        Maximum number of columns allowed. Defaults to 3.

    Returns
    -------
    tuple
        (n_rows, n_cols) for the subplot layout.
    """
    if n_plots <= 1:
        return (1, 1)

    n_cols = min(n_plots, n_cols)
    n_rows = (n_plots + n_cols - 1) // n_cols

    return n_rows, n_cols


def mean_and_var_surface(
    model: Emulator,
    simulator: Simulator,
    variables: list[str],
    output_idx: int = 0,
    quantile: float = 0.5,
    n_points: int = 30,
) -> tuple[TensorLike, TensorLike | None, tuple[TensorLike, ...]]:
    """Create predicted mean and variance on a specified grid for variable subset.

    Create a grid of points varying specified parameters over a specified range,
    while fixing other parameters at a given quantile along their simulation range.

    Parameters
    ----------
    emu: Emulator
        A trained emulator.
    sim: Simulator
        The simulator used to define parameter ranges.
    variables: list[str]
        A list of parameter names to vary.
    output_idx: int,
        The index of the output to return.
    quantile: float
        The quantile at which to fix other parameters. Defaults to 0.5 (median).
    n_points: int
        Number of grid points per variable. Defaults to 30. Higher values increase
        resolution but also computation time.

    Returns
    -------
    mean: TensorLike
        The predicted mean on the grid.
    var: TensorLike
        The predicted variance on the grid.
    grid: list[TensorLike]
        The grid of parameter values used for predictions.

    """
    # Determine which parameters to vary and which to fix
    grid_params = {}
    fixed_params = {}
    for idx, (param_name, param_range) in enumerate(simulator.parameters_range.items()):
        if param_name in variables:
            grid_params[idx] = torch.linspace(param_range[0], param_range[1], n_points)
        else:
            fixed_params[idx] = (
                param_range[1] - param_range[0]
            ) * quantile + param_range[0]

    # Create meshgrid
    grid = torch.meshgrid(*grid_params.values(), indexing="ij")
    x_grid = torch.stack([g.reshape(-1) for g in grid], dim=1)

    def expand_grid(x_grid, fixed_params, grid_params):
        # Fill in fixed parameters
        n_params = len(fixed_params) + len(grid_params)
        x_expanded = torch.empty((x_grid.shape[0], n_params), dtype=torch.float32)
        grid_idx = 0
        for idx in range(n_params):
            if idx in grid_params:
                x_expanded[:, idx] = x_grid[:, grid_idx]
                grid_idx += 1
            else:
                x_expanded[:, idx] = fixed_params[idx]
        return x_expanded

    mean, var = model.predict_mean_and_variance(
        expand_grid(x_grid, fixed_params, grid_params)
    )
    # Subset to specified output_idx
    var = var[:, output_idx : output_idx + 1] if var is not None else var
    return mean[:, output_idx : output_idx + 1], var, grid


def _plot_2d_slice_with_fixed_params(
    mean: TensorLike,
    var: TensorLike | None,
    grid: tuple[TensorLike, ...],
    param_names: list[str],
    lower: float | None,
    upper: float | None,
    fixed_params_info=None,
) -> tuple[Figure, np.ndarray]:
    """Plot 2D slices when other parameters are held constant.

    This works when you have a 2D grid with other parameters fixed.
    """
    ncols = 2 if var is not None else 1
    fig, axs = plt.subplots(
        1,
        ncols,
        figsize=(10, 4) if var is not None else (5, 4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # Get grid dimensions
    n_points_x, n_points_y = grid[0].shape

    # Get coordinate ranges
    x_min, x_max = grid[0].min(), grid[0].max()
    y_min, y_max = grid[1].min(), grid[1].max()

    # Reshape predictions to match 2D grid
    mean_2d = mean.reshape(n_points_x, n_points_y)
    var_2d = var.reshape(n_points_x, n_points_y) if var is not None else None

    # Plot mean
    ax = axs[0, 0]
    im0 = ax.imshow(
        mean_2d.T,  # Transpose for correct orientation
        origin="lower",
        aspect="auto",
        extent=[x_min, x_max, y_min, y_max],
        cmap="viridis",
        vmin=lower,
        vmax=upper,
    )
    ax.set_title("predicted mean")
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    fig.colorbar(im0, ax=ax)

    if var_2d is not None:
        # Plot variance
        ax = axs[0, 1]
        im1 = ax.imshow(
            var_2d.T,
            origin="lower",
            aspect="auto",
            extent=[x_min, x_max, y_min, y_max],
            cmap="magma",
        )
        ax.set_title("predicted variance")
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        fig.colorbar(im1, ax=ax)

    # Add fixed parameters info if provided
    if fixed_params_info:
        fig.suptitle(f"Fixed parameters: {fixed_params_info}", y=1.02)

    plt.tight_layout()
    return fig, axs


def create_and_plot_slice(
    model: Emulator,
    simulator: Simulator,
    param_pair: tuple[int, int],
    output_idx: int = 0,
    vmin: float | None = None,
    vmax: float | None = None,
    quantile: float = 0.5,
    n_points: int = 50,
) -> tuple[Figure, np.ndarray]:
    """Create a 2D slice for any pair of parameters.

    Parameters
    ----------
    model: Emulator
        A trained emulator.
    simulator: Simulator
        The simulator used to define parameter ranges.
    param_pair: tuple[int, int]
        A list of two parameter indices.
    output_idx: int
        The output index to plot the surface of.
    vmin: float | None
        Minimum value for the mean plot color scale.
    vmax: float | None
        Maximum value for the mean plot color scale.
    quantile: float
        The quantile at which to fix other parameters. Defaults to 0.5 (median).
    n_points: int
        Number of grid points per parameter. Defaults to 50.

    Returns
    -------
    mean: TensorLike
        The predicted mean on the grid.
    var: TensorLike
        The predicted variance on the grid.
    grid: list[TensorLike]
        The grid points for the two varying parameters.
    """
    param_pair_names = [
        simulator.param_names[param_pair[0]],
        simulator.param_names[param_pair[1]],
    ]

    # Get the predicted mean and var across a grid for non-fixed params
    mean, var, grid = mean_and_var_surface(
        model,
        simulator,
        variables=param_pair_names,
        output_idx=output_idx,
        quantile=quantile,
        n_points=n_points,
    )

    # Get the names of other fixed parameters
    all_params = list(simulator.parameters_range.keys())
    fixed_params = [p for p in all_params if p not in param_pair_names]

    fig, ax = _plot_2d_slice_with_fixed_params(
        mean,
        var,
        grid,
        param_pair_names,
        vmin,
        vmax,
        fixed_params_info=f"{', '.join(fixed_params)} at {quantile:.1f} quantile"
        if len(fixed_params) > 0
        else "None",
    )
    return fig, ax
