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

    org_points_color = "Goldenrod"
    pred_points_color = "#6A5ACD"
    pred_line_color = "#6A5ACD"

    assert ax is not None, "ax must be provided"
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

    # Add legend
    ax.legend(
        handles[-2:],
        ["data", "pred."],
        loc="best",
        handletextpad=0,
        columnspacing=0,
        ncol=2,
    )

    ax.text(
        0.05,
        0.05,
        f"R\u00b2 = {r2_score:.2f}",
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
