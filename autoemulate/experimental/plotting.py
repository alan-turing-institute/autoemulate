import matplotlib.pyplot as plt
import numpy as np
from IPython.core.getipython import get_ipython
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from autoemulate.experimental.types import NumpyLike


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


def plot_xy(  # noqa: PLR0913
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
