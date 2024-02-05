import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PredictionErrorDisplay


def validate_inputs(cv_results, y, model_name):
    """Validates the inputs for plotting.

    Parameters
    ----------
    cv_results : dict
        A list of cross-validation results for each model.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
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

    if y.ndim > 1:
        raise ValueError("Multi-output can't be plotted yet.")


def plot_single_fold(
    cv_results,
    X,
    y,
    model_name,
    fold_index,
    ax,
    plot_type="actual_vs_predicted",
    annotation=" ",
):
    """Plots a single fold for a given model.

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
    annotation : str, optional
        The type of plot to draw:
        “actual_vs_predicted” draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
        “residual_vs_predicted” draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).

    """
    test_indices = cv_results[model_name]["indices"]["test"][fold_index]

    true_values = y[test_indices]
    predicted_values = cv_results[model_name]["estimator"][fold_index].predict(
        X[test_indices]
    )

    display = PredictionErrorDisplay.from_predictions(
        y_true=true_values, y_pred=predicted_values, kind=plot_type, ax=ax
    )
    ax.set_title(f"{model_name} - {annotation}: {fold_index}")


def plot_best_fold_per_model(
    cv_results, X, y, n_cols=4, plot_type="actual_vs_predicted", figsize=None
):
    """Plots the best fold for each model in cv_results.

    Parameters
    ----------
    cv_results : dict
        A list of cross-validation results for each model.
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
    n_cols : int, optional
        The number of columns in the plot. Default is 4.
    plot_type : str, optional
        The type of plot to draw:
        “actual_vs_predicted” or “residual_vs_predicted”.
    figsize : tuple, optional
        Overrides the default figure size.
    """

    n_models = len(cv_results)
    n_rows = int(np.ceil(n_models / n_cols))

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    plt.figure(figsize=figsize)

    if n_models == 1:
        axes = [axes]
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
        )
    plt.tight_layout()
    plt.show()


def plot_model_folds(
    cv_results,
    X,
    y,
    model_name,
    n_cols=5,
    plot_type="actual_vs_predicted",
    figsize=None,
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
    plot_type : str, optional
        The type of plot to draw:
        “actual_vs_predicted” or “residual_vs_predicted”.
    figsize : tuple, optional
        Overrides the default figure size.
    """

    n_folds = len(cv_results[model_name]["estimator"])
    n_rows = int(np.ceil(n_folds / n_cols))

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    plt.figure(figsize=figsize)

    if n_folds == 1:
        axes = [axes]
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
        )
    plt.tight_layout()
    plt.show()


def plot_results(
    cv_results,
    X,
    y,
    model_name=None,
    n_cols=4,
    plot_type="actual_vs_predicted",
    figsize=None,
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
        The number of columns in the plot. Default is 4.
    plot_type : str, optional
        The type of plot to draw:
        “actual_vs_predicted” draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
        “residual_vs_predicted” draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).
    figsize : tuple, optional
        Overrides the default figure size.
    """

    validate_inputs(cv_results, y, model_name)

    if model_name:
        plot_model_folds(cv_results, X, y, model_name, n_cols, plot_type, figsize)
    else:
        plot_best_fold_per_model(cv_results, X, y, n_cols, plot_type, figsize)
