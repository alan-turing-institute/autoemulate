import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PredictionErrorDisplay


def validate_inputs(cv_results, y, model_name):
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
):
    test_indices = cv_results[model_name]["indices"]["test"][fold_index]

    true_values = y[test_indices]
    predicted_values = cv_results[model_name]["estimator"][fold_index].predict(
        X[test_indices]
    )

    display = PredictionErrorDisplay.from_predictions(
        y_true=true_values, y_pred=predicted_values, kind=plot_type, ax=ax
    )
    ax.set_title(f"{model_name} - Best CV-fold: {fold_index}")


def plot_best_fold_per_model(
    cv_results, X, y, n_cols=4, plot_type="actual_vs_predicted"
):
    n_models = len(cv_results)
    n_rows = int(np.ceil(n_models / n_cols))

    # figure size
    plt.figure(figsize=(4 * n_cols, 3 * n_rows))

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
        )
    plt.tight_layout()
    plt.show()


def plot_results(
    cv_results, X, y, model_name=None, n_cols=4, plot_type="actual_vs_predicted"
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
        The name of the model to plot. If None, all models will be plotted.
    n_cols : int, optional
        The number of columns in the plot. Default is 4.
    plot_type : str, optional
        The type of plot to draw:
        “actual_vs_predicted” draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
        “residual_vs_predicted” draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).
    """

    validate_inputs(cv_results, y, model_name)

    if model_name:
        plot_model_folds(cv_results, X, y, model_name)
    else:
        plot_best_fold_per_model(cv_results, X, y, n_cols, plot_type)


def plot_model_folds(cv_results, X, y, model_name, n_cols=5):
    n_folds = len(cv_results[model_name]["estimator"])
    n_rows = int(np.ceil(n_folds / n_cols))

    # figure size
    plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    if n_folds == 1:
        axes = [axes]
    for i in range(n_folds):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        plot_single_fold(cv_results, X, y, model_name, i, ax)
    plt.tight_layout()
    plt.show()


# def _plot_model_folds(cv_results, X, y, model_name):
#     """Plots all folds for a given model.

#     Parameters
#     ----------
#     cv_results: dict
#         A list of cross-validation results for each model.
#     X : array-like, shape (n_samples, n_features)
#         Simulation input.
#     y : array-like, shape (n_samples, n_outputs)
#         Simulation output.
#     model_name : str
#         The name of the model to plot.
#     """
#     # Plot all folds for the specified model
#     n_folds = len(cv_results[model_name]["estimator"])
#     fig, axes = plt.subplots(1, n_folds, figsize=(n_folds * 5, 4))
#     if n_folds == 1:
#         axes = [axes]
#     for i in range(n_folds):
#         _plot_fold(cv_results, X, y, model_name, i, axes[i], annotation=.)
#     plt.tight_layout()
#     plt.show()


# # Plotting
# ax.scatter(true_values, predicted_values)
# ax.set_xlabel("True Values")
# ax.set_ylabel("Predicted Values")
# ax.set_title(f"Model: {model_name}")
# ax.text(
#     0.05,
#     0.95,
#     annotation,
#     transform=ax.transAxes,
#     fontsize=12,
#     verticalalignment="top",
# )


# def plot_results(cv_results, X, y, model_name=None):
#     """
#     Plots the results of cross-validation for a given set of models.

#     Parameters
#     ----------
#     cv_results: dict
#         A list of cross-validation results for each model.
#     X : array-like, shape (n_samples, n_features)
#         Simulation input.
#     y : array-like, shape (n_samples, n_outputs)
#         Simulation output.
#     model_name : (str, optional)
#         The name of the model to plot. If None, all models will be plotted.

#     """
#     # throw error if y is 2d
#     if y.ndim > 1:
#         raise ValueError("Multi-output can't be plotted yet.")
#     # throw error if cv_results is empty
#     if not cv_results:
#         raise ValueError("Run .compare() first.")

#     if model_name:
#         _plot_model_folds(cv_results, X, y, model_name)
#     else:
#         n_models = len(cv_results)
#         n_cols = 4
#         n_rows = int(np.ceil(n_models / n_cols))
#         fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_models * 5, 4))
#         if n_models == 1:
#             axes = [axes]
#         for i, model in enumerate(cv_results):
#             print(f" axes: {axes[i]}")
#             _plot_best_fold(cv_results, X, y, model, axes[i])
#         plt.tight_layout()
#         plt.show()


# def _plot_best_fold(cv_results, X, y, model_name, ax):
#     """Plots the best fold for a given model.

#     Parameters
#     ----------
#     cv_results: dict
#         A list of cross-validation results for each model.
#     X : array-like, shape (n_samples, n_features)
#         Simulation input.
#     y : array-like, shape (n_samples, n_outputs)
#         Simulation output.
#     model_name : str
#         The name of the model to plot.
#     ax : matplotlib.axes.Axes
#         The axes on which to plot the results.
#     """
#     # Find the fold with the best score for the model
#     best_fold_index = np.argmax(cv_results[model_name]["test_r2"])
#     _plot_fold(
#         cv_results,
#         X,
#         y,
#         model_name,
#         best_fold_index,
#         ax,
#         annotation=f"Best Fold: {best_fold_index}",
#     )


# def _plot_fold(cv_results, X, y, model, fold_index, ax, annotation=""):
#     """Plots a single fold for a given model.

#     Parameters
#     ----------
#     cv_results: dict
#         A list of cross-validation results for each model.
#     X : array-like, shape (n_samples, n_features)
#         Simulation input.
#     y : array-like, shape (n_samples, n_outputs)
#         Simulation output.
#     model : str
#         The name of the model to plot.
#     fold_index : int
#         The index of the fold to plot.
#     ax : matplotlib.axes.Axes
#         The axes on which to plot the results.
#     annotation : str
#         The annotation to add to the plot.
#     """
#     # Extract the indices for the test set
#     test_indices = cv_results[model]["indices"]["test"][fold_index]

#     # Extract the true and predicted values
#     true_values = y[test_indices]
#     predicted_values = cv_results[model]["estimator"][fold_index].predict(
#         X[test_indices]
#     )

#     # Plotting
#     ax.scatter(true_values, predicted_values)
#     ax.set_xlabel("True Values")
#     ax.set_ylabel("Predicted Values")
#     ax.set_title(f"Model: {model}")
#     ax.text(
#         0.05,
#         0.95,
#         annotation,
#         transform=ax.transAxes,
#         fontsize=12,
#         verticalalignment="top",
#     )
