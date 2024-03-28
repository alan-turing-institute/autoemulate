import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from tqdm.autonotebook import tqdm

from autoemulate.cross_validate import _run_cv
from autoemulate.cross_validate import _update_scores_df
from autoemulate.data_splitting import _split_data
from autoemulate.emulators import MODEL_REGISTRY
from autoemulate.hyperparam_searching import _optimize_params
from autoemulate.logging_config import _configure_logging
from autoemulate.metrics import METRIC_REGISTRY
from autoemulate.model_processing import _get_and_process_models
from autoemulate.plotting import _plot_model
from autoemulate.plotting import _plot_results
from autoemulate.printing import _print_cv_results
from autoemulate.printing import _print_model_names
from autoemulate.printing import _print_setup
from autoemulate.save import ModelSerialiser
from autoemulate.utils import _get_full_model_name
from autoemulate.utils import _get_model_names_dict
from autoemulate.utils import _redirect_warnings
from autoemulate.utils import get_mean_scores
from autoemulate.utils import get_model_name
from autoemulate.utils import get_short_model_name


class AutoEmulate:
    """
    The AutoEmulate class is the main class of the AutoEmulate package. It is used to set up and compare
    different emulator models on a given dataset. It can also be used to save and load models, and to
    print and plot the results of the comparison.
    """

    def __init__(self):
        """Initializes an AutoEmulate object."""
        self.X = None
        self.y = None
        self.is_set_up = False

    def setup(
        self,
        X,
        y,
        param_search=False,
        param_search_type="random",
        param_search_iters=20,
        test_set_size=0.2,
        scale=True,
        scaler=StandardScaler(),
        reduce_dim=False,
        dim_reducer=PCA(),
        cross_validator=KFold(n_splits=5, shuffle=True),
        n_jobs=None,
        model_subset=None,
        verbose=0,
        log_to_file=False,
    ):
        """Sets up the automatic emulation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.
        param_search : bool
            Whether to perform hyperparameter search over predifined parameter grids.
        param_search_type : str
            Type of hyperparameter search to perform. Can be "grid", "random", or "bayes".
        param_search_iters : int
            Number of parameter settings that are sampled. Only used if
            param_search=True and param_search_type="random".
        scale : bool, default=True
            Whether to scale the data before fitting the models using a scaler.
        scaler : sklearn.preprocessing.StandardScaler
            Scaler to use. Defaults to StandardScaler. Can be any sklearn scaler.
        reduce_dim : bool, default=False
            Whether to reduce the dimensionality of the data before fitting the models.
        dim_reducer : sklearn.decomposition object
            Dimensionality reduction method to use. Can be any method in `sklearn.decomposition`
            with an `n_components` parameter. Defaults to PCA. Specify n_components like so:
            `dim_reducer=PCA(n_components=2)` for choosing 2 principal components, or
            `dim_reducer=PCA(n_components=0.95)` for choosing the number of components that
            explain 95% of the variance. Other methods can have slightly different n_components
            parameter inputs, see the sklearn documentation for more details. Dimension reduction
            is always performed after scaling.
        cross_validator : sklearn.model_selection object
            Cross-validation strategy to use. Defaults to KFold with 5 splits and shuffle=True.
            Can be any object in `sklearn.model_selection` that generates train/test indices.
        n_jobs : int
            Number of jobs to run in parallel. `None` means 1, `-1` means using all processors.
        model_subset : list
            List of models to use. If None, uses all models in MODEL_REGISTRY.
        verbose : int
            Verbosity level. Can be 0, 1, or 2.
        log_to_file : bool
            Whether to log to file.
        """
        self.X, self.y = self._check_input(X, y)
        self.train_idxs, self.test_idxs = _split_data(
            self.X, test_size=test_set_size, random_state=42
        )
        self.model_names = _get_model_names_dict(MODEL_REGISTRY, model_subset)
        self.models = _get_and_process_models(
            MODEL_REGISTRY,
            model_subset=list(self.model_names.keys()),
            y=self.y,
            scale=scale,
            scaler=scaler,
            reduce_dim=reduce_dim,
            dim_reducer=dim_reducer,
        )
        self.metrics = self._get_metrics(METRIC_REGISTRY)
        self.cross_validator = cross_validator
        self.param_search = param_search
        self.search_type = param_search_type
        self.param_search_iters = param_search_iters
        self.scale = scale
        self.scaler = scaler
        self.n_jobs = n_jobs
        self.logger = _configure_logging(log_to_file, verbose)
        self.is_set_up = True
        self.dim_reducer = dim_reducer
        self.reduce_dim = reduce_dim
        self.cv_results = {}

        self.print_setup()

    def _check_input(self, X, y):
        """Checks and possibly converts the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.

        Returns
        -------
        X_converted : array-like, shape (n_samples, n_features)
            Simulation input.
        y_converted : array-like, shape (n_samples, n_outputs)
            Simulation output.
        """
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True, dtype="float32")
        y = y.astype("float32")  # needed for pytorch models
        return X, y

    def _get_metrics(self, METRIC_REGISTRY):
        """
        Get metrics from REGISTRY

        Parameters
        ----------
        METRIC_REGISTRY : dict
            Registry of metrics.

        Returns
        -------
        List[Callable]
            List of metric functions.
        """
        return [metric for metric in METRIC_REGISTRY.values()]

    def compare(self):
        """Compares the emulator models on the data. self.setup() must be run first.

        Returns
        -------
        scores_df : pandas.DataFrame
            Dataframe containing the scores for each model, metric and fold.
        """
        if not self.is_set_up:
            raise RuntimeError("Must run setup() before compare()")

        # Freshly initialise scores dataframe when running compare()
        self.scores_df = pd.DataFrame(
            columns=["model", "short", "metric", "fold", "score"]
        ).astype(
            {
                "model": "object",
                "short": "object",
                "metric": "object",
                "fold": "int64",
                "score": "float64",
            }
        )

        if self.param_search:
            pb_text = "Optimising and cross-validating"
        else:
            pb_text = "Cross-validating"

        with tqdm(total=len(self.models), desc="Initializing") as pbar:
            for i, model in enumerate(self.models):
                model_name = get_model_name(model)
                pbar.set_description(f"{pb_text} {model_name}")

                with _redirect_warnings(self.logger):
                    try:
                        # hyperparameter search
                        if self.param_search:
                            self.models[i] = _optimize_params(
                                X=self.X[self.train_idxs],
                                y=self.y[self.train_idxs],
                                cv=self.cross_validator,
                                model=model,
                                search_type=self.search_type,
                                niter=self.param_search_iters,
                                param_space=None,
                                n_jobs=self.n_jobs,
                                logger=self.logger,
                            )

                        # run cross validation
                        fitted_model, cv_results = _run_cv(
                            X=self.X[self.train_idxs],
                            y=self.y[self.train_idxs],
                            cv=self.cross_validator,
                            model=model,
                            metrics=self.metrics,
                            n_jobs=self.n_jobs,
                            logger=self.logger,
                        )
                    except Exception:
                        self.logger.exception(
                            f"Error cross-validating model {model_name}"
                        )
                        continue
                    finally:
                        pbar.update(1)

                self.models[i] = fitted_model
                self.cv_results[model_name] = cv_results

                # update scores dataframe
                self.scores_df = _update_scores_df(
                    self.scores_df,
                    model_name,
                    self.cv_results[model_name],
                )

        # get best model
        best_model_name, self.best_model = self.get_model(
            rank=1, metric="r2", name=True
        )

        return self.best_model

    def get_model(self, rank=1, metric="r2", name=False):
        """Get a fitted model based on it's rank in the comparison.

        Parameters
        ----------
        rank : int
            Rank of the model to return. Defaults to 1, which is the best model, 2 is the second best, etc.
        metric : str
            Metric to use for determining the best model.
        name : bool
            If True, returns tuple of model name and model. If False, returns only the model.

        Returns
        -------
        model : object
            Model fitted on full data.
        """

        if not hasattr(self, "scores_df"):
            raise RuntimeError("Must run compare() before get_model()")

        # get average scores across folds
        means = get_mean_scores(self.scores_df, metric)
        # get model by rank
        if (rank > len(means)) or (rank < 1):
            raise RuntimeError(f"Rank must be >= 1 and <= {len(means)}")
        chosen_model_name = means.iloc[rank - 1]["model"]

        # get best model:
        for model in self.models:
            if get_model_name(model) == chosen_model_name:
                chosen_model = model
                break

        # check whether the model is fitted
        check_is_fitted(chosen_model)

        if name:
            return chosen_model_name, chosen_model
        return chosen_model

    def refit_model(self, model):
        """Refits a model on the full data.

        Parameters
        ----------
        model : object
            Usually a fitted model.

        Returns
        -------
        model : object
            Refitted model.
        """
        if not hasattr(self, "X"):
            raise RuntimeError("Must run setup() before refit_model()")

        model.fit(self.X, self.y)
        return model

    def refit_models(self):
        """(Re-) fits all models on the full data.

        Returns
        -------
        models : dict
            dict with refitted models.
        """
        if not hasattr(self, "X"):
            raise RuntimeError("Must run setup() before refit_models()")
        for i in range(len(self.models)):
            self.models[i] = self.refit_model(self.models[i])
        return self.models

    def save_model(self, model=None, path=None):
        """Saves model to disk.

        Parameters
        ----------
        model : object, optional
            Model to save. If None, saves the best model.
            If "all", saves all models.
        path : str
            Path to save the model.
        """
        if not hasattr(self, "best_model"):
            raise RuntimeError("Must run compare() before save_model()")
        serialiser = ModelSerialiser()

        if model is None or not isinstance(model, (Pipeline, BaseEstimator)):
            raise ValueError(
                "Model must be provided and should be a scikit-learn pipeline or model"
            )
        serialiser._save_model(model, path)

    def save_models(self, path=None):
        """Saves all models to disk.

        Parameters
        ----------
        path : str
            Directory to save the models.
            If None, saves to the current working directory.
        """
        if not hasattr(self, "best_model"):
            raise RuntimeError("Must run compare() before save_models()")
        serialiser = ModelSerialiser()
        serialiser._save_models(self.models, path)

    def load_model(self, path=None):
        """Loads a model from disk."""
        serialiser = ModelSerialiser()
        if path is None:
            raise ValueError("Filepath must be provided")

        return serialiser._load_model(path)

    def print_model_names(self):
        """Print available models"""
        _print_model_names(self)

    def print_setup(self):
        """Print the setup of the AutoEmulate object."""
        _print_setup(self)

    def print_results(self, model=None, sort_by="r2"):
        """Print cv results.

        Parameters
        ----------
        model : str, optional
            The name of the model to print. If None, the best fold from each model will be printed.
            If a model name is provided, the scores for that model across all folds will be printed.
        sort_by : str, optional
            The metric to sort by. Default is "r2", can also be "rmse".
        """
        model_name = (
            _get_full_model_name(model, self.model_names) if model is not None else None
        )
        _print_cv_results(
            self.models,
            self.scores_df,
            model_name=model_name,
            sort_by=sort_by,
        )

    def plot_results(
        self,
        model=None,
        plot="standard",
        n_cols=3,
        figsize=None,
        output_index=0,
    ):
        """Plots the results of the cross-validation.

        Parameters
        ----------
        model : str
            Name of the model to plot. If None, plots best folds of each models.
            If a model name is specified, plots all folds of that model.
        plot_type : str, optional
            The type of plot to draw:
            “standard” draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
            “residual” draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).
        n_cols : int
            Number of columns in the plot grid.
        figsize : tuple, optional
            Overrides the default figure size.
        output_index : int
            Index of the output to plot. Default is 0.
        """
        model_name = (
            _get_full_model_name(model, self.model_names) if model is not None else None
        )
        _plot_results(
            self.cv_results,
            self.X,
            self.y,
            model_name=model_name,
            n_cols=n_cols,
            plot=plot,
            figsize=figsize,
            output_index=output_index,
        )

    def evaluate_model(self, model=None):
        """
        Evaluates the model on the hold-out set.

        Parameters
        ----------
        model : object
            Fitted model.

        Returns
        -------
        scores_df : pandas.DataFrame
            Dataframe containing the model scores on the test set.
        """
        if model is None:
            raise ValueError("Model must be provided")

        y_pred = model.predict(self.X[self.test_idxs])
        scores = {}
        for metric in self.metrics:
            scores[metric.__name__] = metric(self.y[self.test_idxs], y_pred)

        scores_df = pd.concat(
            [
                pd.DataFrame({"model": [get_model_name(model)]}),
                pd.DataFrame({"short": [get_short_model_name(model)]}),
                pd.DataFrame(scores, index=[0]),
            ],
            axis=1,
        ).round(3)

        return scores_df

    def plot_model(self, model, plot="standard", n_cols=2, figsize=None):
        """Plots the model predictions vs. the true values.

        Parameters
        ----------
        model : object
            Fitted model.
        plot : str, optional
            The type of plot to draw:
            “standard” draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
            “residual” draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).
        n_cols : int, optional
            Number of columns in the plot grid for multi-output. Default is 2.
        """
        _plot_model(
            model,
            self.X[self.test_idxs],
            self.y[self.test_idxs],
            plot,
            n_cols,
            figsize,
        )
