import matplotlib.pyplot as plt
import numpy as np
from IPython.core.getipython import get_ipython

from autoemulate.experimental.types import NumpyLike


def display_figure(fig):
    """
    Display a matplotlib figure appropriately based on the environment
    (Jupyter notebook or terminal).

    Args:
        fig: matplotlib figure object to display

    Returns:
        fig: the input figure object
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


def plot_Xy(  # noqa: PLR0913
    X: NumpyLike,
    y: NumpyLike,
    y_pred: NumpyLike,
    y_variance: NumpyLike | None = None,
    ax=None,
    title: str = "Xy",
    input_index: int | None = None,
    output_index: int | None = None,
    r2_score: float | None = None,
):
    """
    Plots observed and predicted values vs. features.
    """

    # Sort the data
    sort_idx = np.argsort(X).flatten()
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    y_variance_sorted = None
    if y_variance is not None:
        y_variance_sorted = y_variance[sort_idx]
        # TODO: switch to using standard deviation
        # y_std = np.sqrt(y_variance_sorted)

    org_points_color = "Goldenrod"
    pred_points_color = "#6A5ACD"
    pred_line_color = "#6A5ACD"
    ci_color = "lightblue"

    assert ax is not None, "ax must be provided"
    if y_variance_sorted is not None:
        ax.fill_between(
            X_sorted,
            y_pred_sorted - 2 * y_variance_sorted,
            y_pred_sorted + 2 * y_variance_sorted,
            color=ci_color,
            alpha=0.25,
            label="95% Confidence Interval",
        )
    ax.plot(
        X_sorted,
        y_pred_sorted,
        color=pred_line_color,
        label="pred.",
        alpha=0.8,
        linewidth=1,
    )  # , linestyle='--'
    ax.scatter(
        X_sorted,
        y_sorted,
        color=org_points_color,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        label="data",
    )
    ax.scatter(
        X_sorted,
        y_pred_sorted,
        color=pred_points_color,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7,
        label="pred.",
    )

    ax.set_xlabel(f"$X_{input_index}$", fontsize=13)
    ax.set_ylabel(f"$y_{output_index}$", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    # Get the handles and labels for the scatter plots
    handles, _ = ax.get_legend_handles_labels()

    # Add legend and get its bounding box
    lbl = "pred." if y_variance is None else "pred. mean"
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
        bbox = legend.get_window_extent(ax.figure.canvas.get_renderer())
        inv = ax.transAxes.inverted()
        bbox_axes = bbox.transformed(inv)
        # Place the text just below the legend
        x = bbox_axes.x0
        y = bbox_axes.y0 - 0.04  # small offset below legend
        ax.text(
            x,
            y,
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
    """Calculate optimal number of rows and columns for subplots.

    Parameters
    ----------
    n_plots : int
        Number of plots to display
    n_cols : int, optional
        Maximum number of columns allowed, default is 3

    Returns
    -------
    tuple
        (n_rows, n_cols) for the subplot layout
    """
    if n_plots <= 1:
        return (1, 1)

    n_cols = min(n_plots, n_cols)
    n_rows = (n_plots + n_cols - 1) // n_cols

    return n_rows, n_cols
