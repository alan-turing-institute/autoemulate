from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.core.getipython import get_ipython
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from autoemulate.core.types import DistributionLike, GaussianLike, NumpyLike, TensorLike
from autoemulate.emulators.base import Emulator, PyTorchBackend


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
    input_label: str | None = None,
    output_label: str | None = None,
    r2_score: float | None = None,
    error_style: Literal["bars", "fill"] = "bars",
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
    input_label: str | None
        An optional input label to plot.
    output_label: str | None
        An optional output label to plot.
    r2_score: float | None
        An option r2 score to include in the plot legend.
    error_style: Literal["bars", "fill"]
        The style of error representation in the plots. Can be "bars" for error
        bars or "fill" for shaded error regions. Defaults to "bars".
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

    assert ax is not None, "ax must be provided"
    # Scatter plot with error bars for predictions
    if y_std is not None:
        if error_style.lower() not in ["bars", "fill"]:
            msg = "error_style must be one of ['bars', 'fill']"
            raise ValueError(msg)
        if error_style.lower() == "bars":
            ax.errorbar(
                x_sorted,
                y_pred_sorted,
                yerr=2 * y_std,
                fmt="o",
                color=pred_points_color,
                elinewidth=2,
                capsize=3,
                alpha=0.5,
                # use unicode for sigma
                label="pred. (±2\u03c3)",
            )
            ax.scatter(
                x_sorted,
                y_pred_sorted,
                color=pred_points_color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.5,
            )
        else:
            ax.fill_between(
                x_sorted,
                y_pred_sorted - 2 * y_std,
                y_pred_sorted + 2 * y_std,
                color=pred_points_color,
                alpha=0.2,
                label="±2\u03c3",
            )
            ax.plot(
                x_sorted,
                y_pred_sorted,
                color=pred_points_color,
                alpha=0.75,
                label="pred.",
            )
    else:
        ax.scatter(
            x_sorted,
            y_pred_sorted,
            color=pred_points_color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.5,
            label="pred.",
        )
    # Scatter plot for observed data
    ax.scatter(
        x_sorted,
        y_sorted,
        color=org_points_color,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        label="data",
    )

    x_label = input_label if input_label is not None else "x"
    y_label = output_label if output_label is not None else "y"
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    # Get the handles and labels for the scatter plots
    handles, _ = ax.get_legend_handles_labels()

    # Add legend and get its bounding box
    legend = ax.legend(
        loc="best",
        handletextpad=0,
        columnspacing=0,
        ncol=2,
    )

    # Place R² just below the legend
    if legend:
        # Get the bounding box of the legend in axes coordinates
        bbox = legend.get_window_extent(
            ax.figure.canvas.get_renderer()
        )  # pyright: ignore[reportAttributeAccessIssue]
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
    parameters_range: dict[str, tuple[float, float]],
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
    model: Emulator
        A trained emulator.
    parameters_range: dict[str, tuple[float, float]]
        A dictionary specifying the ranges for all input parameters. Keys are parameter
        names and values are tuples of (min, max). The dictionary should be ordered
        equivalently to the order of parameters used to train the model.
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
    for idx, (param_name, param_range) in enumerate(parameters_range.items()):
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
    parameters_range: dict[str, tuple[float, float]],
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
    parameters_range: dict[str, tuple[float, float]]
        A dictionary specifying the ranges for all input parameters. Keys are parameter
        names and values are tuples of (min, max). The dictionary should be ordered
        equivalently to the order of parameters used to train the model.
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
    param_names = list(parameters_range.keys())
    param_pair_names = [param_names[param_pair[0]], param_names[param_pair[1]]]

    # Get the predicted mean and var across a grid for non-fixed params
    mean, var, grid = mean_and_var_surface(
        model,
        parameters_range,
        variables=param_pair_names,
        output_idx=output_idx,
        quantile=quantile,
        n_points=n_points,
    )

    # Get the names of other fixed parameters
    fixed_params = [p for p in param_names if p not in param_pair_names]

    fig, ax = _plot_2d_slice_with_fixed_params(
        mean,
        var,
        grid,
        param_pair_names,
        vmin,
        vmax,
        fixed_params_info=(
            f"{', '.join(fixed_params)} at {quantile:.1f} quantile"
            if len(fixed_params) > 0
            else "None"
        ),
    )
    return fig, ax


def coverage_from_distributions(
    y_pred: DistributionLike,
    y_true: TensorLike,
    levels: list[float] | NumpyLike | TensorLike | None = None,
    n_samples: int = 2000,
    joint: bool = False,
) -> tuple[NumpyLike, NumpyLike]:
    """Compute empirical coverage for a set of nominal confidence levels.

    Parameters
    ----------
    y_pred: DistributionLike
        The emulator predicted distribution.
    y_true: TensorLike
        The true values.
    levels: array-like, optional
        Nominal coverage levels (between 0 and 1). If None, a default grid is
        used. Defaults to None.
    n_samples: int
        Number of Monte-Carlo samples to draw from the predictive
        distribution to compute empirical intervals if analytical quantiles
        are not available.
    joint: bool
        If True and the predictive outputs are multivariate, compute joint
        coverage (i.e., the true vector must lie inside the interval for all
        dimensions). If False (default), compute marginal coverage per output
        dimension and return the mean across data points.

    Returns
    -------
    levels: np.ndarray
        Nominal coverage levels.
    empirical: np.ndarray
        Empirical coverages. Shape is (len(levels), output_dim) when
        `joint=False` and output_dim>1, or (len(levels),) when joint=True or
        output_dim==1.
    """
    if levels is None:
        levels = np.linspace(0.0, 1.0, 51)
    levels = np.asarray(levels)

    # if dist.icdf not available, compute empirical intervals using sample quantiles
    samples = None
    y_dist = None
    if isinstance(y_pred, torch.distributions.Independent) and isinstance(
        y_pred.base_dist, GaussianLike
    ):
        y_dist = y_pred.base_dist
    else:
        samples = y_pred.sample(torch.Size((n_samples,)))

    empirical_list = []
    for p in levels:
        lower_q = (1.0 - p) / 2.0
        upper_q = 1.0 - lower_q

        if y_dist is not None:
            lower = y_dist.icdf(lower_q)
            upper = y_dist.icdf(upper_q)
        else:
            assert samples is not None
            lower = torch.quantile(samples, float(lower_q), dim=0)
            upper = torch.quantile(samples, float(upper_q), dim=0)

        inside = (y_true >= lower) & (y_true <= upper)
        if joint:
            inside_all = inside.all(dim=-1)
            empirical = inside_all.float().mean().item()
        else:
            # marginal per-dim coverage
            empirical = inside.float().mean(dim=0).cpu().numpy()
        empirical_list.append(empirical)

    empirical_arr = np.asarray(empirical_list)

    return levels, empirical_arr


def plot_calibration_from_distributions(
    y_pred: DistributionLike,
    y_true: TensorLike,
    levels: np.ndarray | None = None,
    n_samples: int = 2000,
    joint: bool = False,
    title: str | None = None,
    legend: bool = True,
    figsize: tuple[int, int] | None = None,
):
    """Plot calibration curve(s) given predictive distributions and true values.

    This draws empirical coverage (y-axis) against nominal coverage (x-axis).

    When points lie above or below the diagonal, this indicates that uncertainty
    is respectively being  overestimated or underestimated.

    Parameters
    ----------
    y_pred: DistributionLike
        The emulator predicted distribution.
    y_true: TensorLike
        The true values.
    levels: array-like, optional
        Nominal coverage levels (between 0 and 1). If None, a default grid is
        used.
    n_samples: int
        Number of Monte-Carlo samples to draw from the predictive
        distribution to compute empirical intervals.
    joint: bool
        If True and the predictive outputs are multivariate, compute joint
        coverage (i.e., the true vector must lie inside the interval for all
        dimensions). If False (default), compute marginal coverage per output
        dimension and return the mean across data points.
    title: str | None
        An optional title for the plot. Defaults to None (no title).
    legend: bool
        Whether to display a legend. Defaults to True.
    figsize: tuple[int, int] | None
        The size of the figure to create. If None, a default size is used.
    """
    levels, empirical = coverage_from_distributions(
        y_pred, y_true, levels=levels, n_samples=n_samples, joint=joint
    )

    if figsize is None:
        figsize = (6, 6)
    fig, ax = plt.subplots(figsize=figsize)

    if len(empirical.shape) == 1 or empirical.shape[1] == 1:
        ax.plot(levels, empirical, marker="o", label="empirical")
    else:
        # multiple outputs: plot each dimension
        for i in range(empirical.shape[1]):
            ax.plot(levels, empirical[:, i], marker="o", label=f"$y_{i}$")

    # diagonal reference
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")

    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    if legend:
        ax.legend()

    return fig, ax


def plot_loss(
    model: PyTorchBackend, title: str | None = None, figsize: tuple[int, int] | None = None
):
    """
    Plot the training loss curve for a model using the PyTorch backend.

    This function visualizes the per-epoch training loss stored in the
    model's ``loss_history`` attribute. The model must also expose an
    ``epochs`` attribute; if either attribute is missing, an
    ``AttributeError`` is raised.

    Parameters
    ----------
    model : PyTorchBackend
        A model instance using the PyTorch backend, required to provide
        ``loss_history`` and ``epochs`` attributes.
    title : str, optional
        Title for the plot. If ``None``, no title is added.
    figsize : tuple of int, optional
        Size of the figure as ``(width, height)`` in inches. Defaults to
        ``(6, 6)`` if not provided.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The Axes on which the loss curve is plotted.

    Raises
    ------
    AttributeError
        If the model does not provide ``loss_history`` or ``epochs`` attributes.
    """

    if not hasattr(model, "loss_history"):
        msg = "Emulator does not have a Loss history"
        raise AttributeError(msg)
    
    history = model.loss_history
    
    if figsize is None:
        figsize = (6, 6)

    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(history) + 1), history)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Train Loss")

    if title:
        ax.set_title(title)

    return fig, ax
