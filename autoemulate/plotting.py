import inspect

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PredictionErrorDisplay

from autoemulate.utils import get_model_name


def _validate_inputs(cv_results, model_name):
    """Validates cv_results and model_name for plotting.

    Parameters
    ----------
    cv_results : dict
        A list of cross-validation results for each model.
    model_name : str
        The name of a model to plot.
    """
    if not cv_results:
        raise ValueError("Run .compare() first.")

    if model_name:
        if model_name not in cv_results:
            raise ValueError(
                f"Model {model_name} not found. Available models are: {cv_results.keys()}"
            )


def check_multioutput(y, output_index):
    """Checks if y is multi-output and if the output_index is valid."""
    if y.ndim > 1:
        if (output_index > y.shape[1] - 1) | (output_index < 0):
            raise ValueError(
                f"Output index {output_index} is out of range. The index should be between 0 and {y.shape[1] - 1}."
            )
        print(
            f"""Multiple outputs detected. Plotting the output variable with index {output_index}. 
To plot other outputs, set `output_index` argument to the desired index."""
        )


def _plot_single_fold(
    cv_results,
    X,
    y,
    model_name,
    fold_index,
    ax,
    plot="standard",
    annotation=" ",
    output_index=0,
    input_index=0,
):
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
    plot : str, optional
        The type of plot to draw:
        “standard” draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
        “residual” draws the residuals, i.e. difference between observed and predicted values,
        (y-axis) vs. the predicted values (x-axis).
    annotation : str, optional
        The annotation to add to the plot title. Default is an empty string.
    output_index : int, optional
        The index of the output to plot. Default is 0.
    input_index : int, optional
        The index of the input variable to plot. Default is 0.
    """
    # get cv fold test indices
    test_indices = cv_results[model_name]["indices"]["test"][fold_index]
    X_test = X[test_indices]
    y_test = y[test_indices]
    # get model trained on cv fold train indices
    model = cv_results[model_name]["estimator"][fold_index]
    # should we return and plot uncertainty?
    predict_params = inspect.signature(model.named_steps["model"].predict).parameters
    if "return_std" in predict_params:
        y_test_pred, y_test_std = model.predict(X_test, return_std=True)
    else:
        y_test_pred = model.predict(X_test)
        y_test_std = None

    # if y is multi-output, we need to select the correct column
    if y.ndim > 1:
        y_test = y_test[:, output_index]
        y_test_pred = y_test_pred[:, output_index]
        if y_test_std is not None:
            y_test_std = y_test_std[:, output_index]

    match plot:
        case "standard":
            plot_type = "actual_vs_predicted"
        case "residual":
            plot_type = "residual_vs_predicted"
        case "Xy":
            plot_type = "Xy"
        case _:
            ValueError(f"Invalid plot type: {plot}")

    if plot_type == "Xy":
        # if X is multi-dimensional, we need to select the correct column
        if X.ndim > 1:
            X_test = X_test[:, input_index]
        title_suffix = f"{annotation}: {fold_index}"
        _plot_Xy(
            X_test,
            y_test,
            y_test_pred,
            y_test_std,
            ax,
            title=f"{model_name} - {title_suffix}",
        )
    else:
        display = PredictionErrorDisplay.from_predictions(
            y_true=y_test, y_pred=y_test_pred, kind=plot_type, ax=ax
        )
        title_suffix = f"{annotation}: {fold_index}"
        ax.set_title(f"{model_name} - {title_suffix}")


def _plot_best_fold_per_model(
    cv_results,
    X,
    y,
    n_cols=3,
    plot="standard",
    figsize=None,
    output_index=0,
):
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
    plot : str, optional
        The type of plot to draw:
        “standard" or "residual”.
    figsize : tuple, optional
        Width, height in inches. Overrides the default figure size.
    output_index : int, optional
        The index of the output to plot. Default is 0.
    """

    n_models = len(cv_results)
    n_rows = int(np.ceil(n_models / n_cols))

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    plt.figure(figsize=figsize)

    for i, model_name in enumerate(cv_results):
        best_fold_index = np.argmax(cv_results[model_name]["test_r2"])
        ax = plt.subplot(n_rows, n_cols, i + 1)
        _plot_single_fold(
            cv_results,
            X,
            y,
            model_name,
            best_fold_index,
            ax,
            plot=plot,
            annotation="Best CV-fold",
            output_index=output_index,
        )
    plt.tight_layout()
    plt.show()


def _plot_model_folds(
    cv_results,
    X,
    y,
    model_name,
    n_cols=3,
    plot="standard",
    figsize=None,
    output_index=0,
):
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
    plot : str, optional
        The type of plot to draw:
        “standard” or “residual”.
    figsize : tuple, optional
        Overrides the default figure size.
    output_index : int, optional
        The index of the output to plot. Default is 0.
    """

    n_folds = len(cv_results[model_name]["estimator"])
    n_rows = int(np.ceil(n_folds / n_cols))

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    plt.figure(figsize=figsize)

    for i in range(n_folds):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        _plot_single_fold(
            cv_results,
            X,
            y,
            model_name,
            i,
            ax,
            plot,
            annotation="CV-fold",
            output_index=output_index,
        )
    plt.tight_layout()
    plt.show()


def _plot_results(
    cv_results,
    X,
    y,
    model_name=None,
    n_cols=3,
    plot="standard",
    figsize=None,
    output_index=0,
):
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
    plot : str, optional
        The type of plot to draw:
        “standard” draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
        “residual” draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).
    figsize : tuple, optional
        Overrides the default figure size.
    output_index : int, optional
        For multi-output: Index of the output variable to plot.
    """

    _validate_inputs(cv_results, model_name)
    check_multioutput(y, output_index)

    if model_name:
        _plot_model_folds(
            cv_results,
            X,
            y,
            model_name,
            n_cols,
            plot,
            figsize,
            output_index,
        )
    else:
        _plot_best_fold_per_model(cv_results, X, y, n_cols, plot, figsize, output_index)


def _plot_model(model, X, y, plot="standard", n_cols=2, figsize=None):
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
                    y_true=y[:, i], y_pred=y_pred[:, i], kind=plot_type, ax=axs[i]
                )
                axs[i].set_title(f"{get_model_name(model)} - Test Set - Output {i+1}")

        # Hide any unused subplots if n_cols * n_rows > n_outputs
        for ax in axs[n_outputs:]:
            ax.set_visible(False)

    plt.show()


def _plot_Xy(X, y, y_pred, y_std=None, ax=None, title="Xy"):
    # Sort the data
    sort_idx = np.argsort(X).flatten()
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    if y_std is not None:
        y_std_sorted = y_std[sort_idx]

    org_points_color = "#4B0082"
    pred_points_color = "#00CED1"
    pred_line_color = "#00CED1"
    ci_color = "#9DC183"

    ax.scatter(
        X_sorted, y_sorted, color=org_points_color, alpha=0.7, label="data", s=50
    )
    ax.scatter(
        X_sorted, y_pred_sorted, color=pred_points_color, alpha=1, label="pred.", s=10
    )
    if y_std is not None:
        ax.fill_between(
            X_sorted,
            y_pred_sorted - 2 * y_std_sorted,
            y_pred_sorted + 2 * y_std_sorted,
            color=ci_color,
            alpha=0.5,
            label="95% Confidence Interval",
        )
    ax.plot(X_sorted, y_pred_sorted, color=pred_line_color, label="pred.")

    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add legend
    if y_std is not None:
        ax.legend(["data", "pred.(±2σ)"], loc="best")
    else:
        ax.legend(["data", "pred."], loc="best")
    # Calculate R2 score
    r2 = 1 - np.sum((y_sorted - y_pred_sorted) ** 2) / np.sum(
        (y_sorted - np.mean(y_sorted)) ** 2
    )
    ax.text(
        0.05,
        0.95,
        f"R2 score: {r2:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )
