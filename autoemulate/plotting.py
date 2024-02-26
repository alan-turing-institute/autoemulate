import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PredictionErrorDisplay

from .types import TYPE_CHECKING
from autoemulate.utils import get_model_name

if TYPE_CHECKING:
    from .types import ArrayLike, Literal, Optional

    PlotTypes = Literal["actual_vs_predicted", "residual_vs_predicted"]


def validate_inputs(cv_results: dict, model_name: str) -> None:
    """Validates cv_results and model_name for plotting.

    Parameters
    ----------
    cv_results : dict
        A list of cross-validation results for each model.
    model_name : str
        The name of a model to plot.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If cv_results is empty.
        If model_name is not in cv_results.
    """
    if not cv_results:
        raise ValueError("Run .compare() first.")

    if model_name:
        if model_name not in cv_results:
            raise ValueError(
                f"Model {model_name} not found. Available models are: {cv_results.keys()}"
            )


def check_multioutput(y: ArrayLike, output_index: int) -> None:
    """Checks if y is multi-output and if the output_index is valid.

    Parameters
    ----------
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
    output_index : int
        The index of the output to plot.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If output_index is out of range.
    """
    if y.ndim > 1:
        if (output_index > y.shape[1] - 1) | (output_index < 0):
            raise ValueError(
                f"Output index {output_index} is out of range. The index should be between 0 and {y.shape[1] - 1}."
            )
        print(
            f"""Multiple outputs detected. Plotting the output variable with index {output_index}. 
To plot other outputs, set `output_index` argument to the desired index."""
        )


# TODO: Should X be MatrixLike?
def plot_single_fold(
    cv_results: dict,
    X: ArrayLike,
    y: ArrayLike,
    model_name: str,
    fold_index: int,
    ax: plt.Axes,
    plot_type: PlotTypes = "actual_vs_predicted",
    annotation: str = " ",
    output_index: int = 0,
) -> None:
    """Plots a single cv fold for a given model.

    Parameters
    ----------
    cv_results : dict
        A list of cross-validation results for each model.
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
    model_name : str
        The name of the model to plot.
    fold_index : int
        The index of the fold to plot.
    ax : matplotlib.axes.Axes
        The axes on which to plot the results.
    plot_type : str, optional
        The type of plot to draw:
        “actual_vs_predicted” draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
        “residual_vs_predicted” draws the residuals, i.e. difference between observed and predicted values,
        (y-axis) vs. the predicted values (x-axis).
    annotation : str, optional
        The annotation to add to the plot title. Default is an empty string.
    output_index : int, optional
        The index of the output to plot. Default is 0.

    Returns
    -------
    None
    """
    test_indices = cv_results[model_name]["indices"]["test"][fold_index]

    true_values = y[test_indices]

    predicted_values = cv_results[model_name]["estimator"][fold_index].predict(
        X[test_indices]
    )

    # if y is multi-output, we need to select the correct column
    if y.ndim > 1:
        true_values = true_values[:, output_index]
        predicted_values = predicted_values[:, output_index]
    # plot
    display = PredictionErrorDisplay.from_predictions(
        y_true=true_values, y_pred=predicted_values, kind=plot_type, ax=ax
    )
    title_suffix = f"{annotation}: {fold_index}"
    ax.set_title(f"{model_name} - {title_suffix}")


# TODO: Should X be MatrixLike?
def plot_best_fold_per_model(
    cv_results: dict,
    X: ArrayLike,
    y: ArrayLike,
    n_cols: int = 3,
    plot_type: PlotTypes = "actual_vs_predicted",
    figsize: Optional[tuple[int, int]] = None,
    output_index: int = 0,
) -> None:
    """Plots results of the best (highest R^2) cv-fold for each model in cv_results.

    Parameters
    ----------
    cv_results : dict
        A list of cross-validation results for each model.
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
    n_cols : int, optional
        The number of columns in the plot. Default is 3.
    plot_type : str, optional
        The type of plot to draw:
        “actual_vs_predicted” or “residual_vs_predicted”.
    figsize : tuple, optional
        Width, height in inches. Overrides the default figure size.
    output_index : int, optional
        The index of the output to plot. Default is 0.

    Returns
    -------
    None
    """

    n_models = len(cv_results)
    n_rows = int(np.ceil(n_models / n_cols))

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    plt.figure(figsize=figsize)

    for i, model_name in enumerate(cv_results):
        best_fold_index = np.argmax(cv_results[model_name]["test_r2"])
        ax = plt.subplot(n_rows, n_cols, i + 1)
        plot_single_fold(
            cv_results,
            X,
            y,
            model_name,
            best_fold_index,
            ax,
            plot_type=plot_type,
            annotation="Best CV-fold",
            output_index=output_index,
        )
    plt.tight_layout()
    plt.show()


# TODO: Should X be MatrixLike?
def plot_model_folds(
    cv_results: dict,
    X: ArrayLike,
    y: ArrayLike,
    model_name: str,
    n_cols: int = 3,
    plot_type: PlotTypes = "actual_vs_predicted",
    figsize: Optional[tuple[int, int]] = None,
    output_index: int = 0,
) -> None:
    """Plots all the folds for a given model.

    Parameters
    ----------
    cv_results : dict
        A list of cross-validation results for each model.
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
    model_name : str
        The name of the model to plot.
    n_cols : int, optional
        The number of columns in the plot. Default is 5.
    plot_type : str, optional
        The type of plot to draw:
        “actual_vs_predicted” or “residual_vs_predicted”.
    figsize : tuple, optional
        Overrides the default figure size.
    output_index : int, optional
        The index of the output to plot. Default is 0.

    Returns
    -------
    None
    """

    n_folds = len(cv_results[model_name]["estimator"])
    n_rows = int(np.ceil(n_folds / n_cols))

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    plt.figure(figsize=figsize)

    for i in range(n_folds):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        plot_single_fold(
            cv_results,
            X,
            y,
            model_name,
            i,
            ax,
            plot_type,
            annotation="CV-fold",
            output_index=output_index,
        )
    plt.tight_layout()
    plt.show()


# TODO: Should X be MatrixLike?
def _plot_results(
    cv_results: dict,
    X: ArrayLike,
    y: ArrayLike,
    model_name: Optional[str] = None,
    n_cols: int = 3,
    plot_type: PlotTypes = "actual_vs_predicted",
    figsize: Optional[tuple[int, int]] = None,
    output_index: int = 0,
) -> None:
    """Plots the results of cross-validation.

    Parameters
    ----------
    cv_results : dict
        A list of cross-validation results for each model.
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
    model_name : (str, optional)
        The name of the model to plot. If None, the best (largest R^2) fold for each model will be plotted.
    n_cols : int, optional
        The number of columns in the plot. Default is 3.
    plot_type : str, optional
        The type of plot to draw:
        “actual_vs_predicted” draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
        “residual_vs_predicted” draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).
    figsize : tuple, optional
        Overrides the default figure size.
    output_index : int, optional
        For multi-output: Index of the output variable to plot.
    """

    validate_inputs(cv_results, model_name)
    check_multioutput(y, output_index)

    if model_name:
        plot_model_folds(
            cv_results,
            X,
            y,
            model_name,
            n_cols,
            plot_type,
            figsize,
            output_index,
        )
    else:
        plot_best_fold_per_model(
            cv_results, X, y, n_cols, plot_type, figsize, output_index
        )


# TODO: add model type, is Pipeline correct?
def _plot_model(
    model,
    X: ArrayLike,
    y: ArrayLike,
    plot: Literal["standard", "residual"] = "standard",
    n_cols: int = 2,
    figsize: Optional[tuple[int, int]] = None,
):
    """Plots the model predictions vs. the true values.

    Parameters
    ----------
    model : object
        A fitted model.
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
    plot : str, optional
        The type of plot to draw:
        “standard” draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
        “residual” draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).
    n_cols : int, optional
        The number of columns in the plot. Default is 2.
    figsize : tuple, optional
        Overrides the default figure size.
    """

    match plot:
        case "standard":
            plot_type = "actual_vs_predicted"
        case "residual":
            plot_type = "residual_vs_predicted"
        case _:
            ValueError(f"Invalid plot type: {plot}")

    # figsize
    if figsize is None:
        if y.ndim == 1 or y.shape[1] == 1:
            figsize = (6, 4)
        else:  # Dynamic calculation for multi-output
            n_outputs = y.shape[1]
            n_rows = np.ceil(n_outputs / n_cols).astype(int)
            figsize = (4 * n_cols, 4 * n_rows)
    # predictions
    y_pred = model.predict(X)

    if y.ndim == 1 or y.shape[1] == 1:  # single output
        _, ax = plt.subplots(figsize=figsize)
        display = PredictionErrorDisplay.from_predictions(
            y_true=y, y_pred=y_pred, kind=plot_type, ax=ax
        )
        ax.set_title(f"{get_model_name(model)} - Test Set")
    else:  # Multi-output
        n_outputs = y.shape[1]
        n_rows = np.ceil(n_outputs / n_cols).astype(int)
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=figsize, constrained_layout=True
        )
        axs = axs.flatten()

        for i in range(n_outputs):
            if i < len(
                axs
            ):  # Check to avoid index error if n_cols * n_rows > n_outputs
                display = PredictionErrorDisplay.from_predictions(
                    y_true=y[:, i],
                    y_pred=y_pred[:, i],
                    kind=plot_type,
                    ax=axs[i],
                )
                axs[i].set_title(f"{get_model_name(model)} - Test Set - Output {i+1}")

        # Hide any unused subplots if n_cols * n_rows > n_outputs
        for ax in axs[n_outputs:]:
            ax.set_visible(False)

    plt.show()
