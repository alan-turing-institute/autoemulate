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
from autoemulate.cross_validate import _sum_cv
from autoemulate.cross_validate import _sum_cvs
from autoemulate.data_splitting import _split_data
from autoemulate.emulators import model_registry
from autoemulate.hyperparam_searching import _optimize_params
from autoemulate.logging_config import _configure_logging
from autoemulate.metrics import METRIC_REGISTRY
from autoemulate.model_processing import _process_models
from autoemulate.plotting import _plot_cv
from autoemulate.plotting import _plot_model
from autoemulate.printing import _print_setup
from autoemulate.save import ModelSerialiser
from autoemulate.sensitivity_analysis import perform_sobol_analysis
from autoemulate.sensitivity_analysis import plot_sensitivity_indices
from autoemulate.utils import _ensure_2d
from autoemulate.utils import _get_full_model_name
from autoemulate.utils import _redirect_warnings
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
        models=None,
        verbose=0,
        log_to_file=False,
        print_setup=True,
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
            Type of hyperparameter search to perform. Currently only "random".
        param_search_iters : int
            Number of parameter settings that are sampled. Only used if
            param_search=True and param_search_type="random".
        scale : bool, default=True
            Whether to scale features/parameters in X before fitting the models using a scaler.
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
        models : list
            str or list of model names. If None, uses a set of core models.
        verbose : int
            Verbosity level. Can be 0, 1, or 2.
        log_to_file : bool
            Whether to log to file.
        print_setup : bool
            Whether to print the setup of the AutoEmulate object.
        """
        self.model_registry = model_registry
        self.X, self.y = self._check_input(X, y)
        self.test_set_size = test_set_size
        self.train_idxs, self.test_idxs = _split_data(
            self.X, test_size=self.test_set_size, random_state=42
        )
        self.model_names = self.model_registry.get_model_names(models, is_core=True)
        self.models = _process_models(
            model_registry=self.model_registry,
            models=list(self.model_names.keys()),
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

        if print_setup:
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
        self.best_model : object
            Best performing model fitted on full data.
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
                            model=self.models[i],
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

        # get best model
        self.best_model = self.get_model(rank=1, metric="r2")

        return self.best_model

    def get_model(self, name=None, rank=1, metric="r2"):
        """Get a fitted model based on its name or rank in the comparison.

        Parameters
        ----------
        name : str
            Name of the model to return.
        rank : int
            Rank of the model to return. Defaults to 1, which is the best model, 2 is the second best, etc.
        metric : str
            Metric to use for determining the best model.

        Returns
        -------
        model : object
            Model fitted on full data.
        """

        # get model by name
        if name is not None:
            if not isinstance(name, str):
                raise ValueError("Name must be a string")
            for model in self.models:
                if get_model_name(model) == name or get_short_model_name(model) == name:
                    return model
            raise ValueError(f"Model {name} not found")

        # check that comparison has been run
        if not hasattr(self, "cv_results") and name is None:
            raise RuntimeError("Must run compare() before get_model()")

        # get model by rank
        means = _sum_cvs(self.cv_results, metric)

        if (rank > len(means)) or (rank < 1):
            raise RuntimeError(f"Rank must be >= 1 and <= {len(means)}")
        chosen_model_name = means.iloc[rank - 1]["model"]

        for model in self.models:
            if get_model_name(model) == chosen_model_name:
                chosen_model = model
                break

        # check_is_fitted(chosen_model)
        return chosen_model

    def refit(self, model=None):
        """Refits model on full data.

        Parameters
        ----------
        model : model to refit.

        Returns
        -------
        model : object
            Refitted model.
        """
        if not hasattr(self, "X"):
            raise RuntimeError("Must run setup() before refit()")
        if model is None:
            raise ValueError("Model must be provided")
        else:
            if not isinstance(model, BaseEstimator):
                raise ValueError(
                    "Model must be provided and should be a scikit-learn estimator"
                )
        model.fit(self.X, self.y)
        return model

    def save(self, model=None, path=None):
        """Saves model to disk.

        Parameters
        ----------
        model : sklearn model, optional
            Model to save. If None, saves the model with the best
            average cv score.
        path : str
            Path to save the model.
        """
        if not hasattr(self, "best_model"):
            raise RuntimeError("Must run compare() before save()")

        serialiser = ModelSerialiser(self.logger)
        if model is None:
            serialiser._save_model(self.best_model, path)
        else:
            if not isinstance(model, BaseEstimator):
                raise ValueError(
                    "Model must be provided and should be a scikit-learn estimator"
                )
        serialiser._save_model(model, path)

    def load(self, path=None):
        """Loads a model from disk.

        Parameters
        ----------
        path : str
            Path to model.
        """
        if path is None:
            raise ValueError("Path must be provided")
        serialiser = ModelSerialiser(self.logger)
        return serialiser._load_model(path)

    def print_setup(self):
        """Print the setup of the AutoEmulate object."""
        _print_setup(self)

    def summarise_cv(self, model=None, sort_by="r2"):
        """Summarise cv results.

        Parameters
        ----------
        model : str, optional
            Name of the model to get cv results for. If None, summarises results for all models.
        sort_by : str, optional
            The metric to sort by. Default is "r2", can also be "rmse".

        Returns
        -------
        pandas.DataFrame
            DataFrame summarising cv results.
        """
        model_name = (
            _get_full_model_name(model, self.model_names) if model is not None else None
        )

        if model_name is None:
            cv = _sum_cvs(self.cv_results, sort_by)
        else:
            cv = _sum_cv(self.cv_results[model_name])

        return cv

    summarize_cv = summarise_cv  # alias

    def plot_cv(
        self,
        model=None,
        style="Xy",
        n_cols=3,
        figsize=None,
        output_index=0,
        input_index=0,
    ):
        """Plots the results of the cross-validation.

        Parameters
        ----------
        model : str
            Name of the model to plot. If None, plots best folds of each models.
            If a model name is specified, plots all folds of that model.
        style : str, optional
            The type of plot to draw:
            "Xy" observed and predicted values vs. features, including 2σ error bands where available (default).
            "actual_vs_predicted" draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
            "residual_vs_predicted" draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).
        n_cols : int
            Number of columns in the plot grid.
        figsize : tuple, optional
            Overrides the default figure size.
        output_index : int
            Index of the output to plot. Default is 0.
        input_index : int
            Index of the input to plot. Default is 0.
        """
        model_name = (
            _get_full_model_name(model, self.model_names) if model is not None else None
        )
        figure = _plot_cv(
            self.cv_results,
            self.X,
            self.y,
            model_name=model_name,
            n_cols=n_cols,
            style=style,
            figsize=figsize,
            output_index=output_index,
            input_index=input_index,
        )
        return figure

    def evaluate(self, model=None, multioutput="uniform_average"):
        """
        Evaluates the model on the test set.

        Parameters
        ----------
        model : object
            Fitted model
        multioutput : str, optional
            Defines aggregating of multiple output scores.
            'raw_values' : returns scores for each target
            'uniform_average' : scores are averaged uniformly
            'variance_weighted' : scores are averaged weighted by their individual variances

        Returns
        -------
        scores_df : pandas.DataFrame
            Dataframe containing the model scores on the test set.
        """
        if model is None:
            raise ValueError("Model must be provided")
        if not isinstance(model, BaseEstimator):
            raise ValueError("Model should be a fitted model")

        y_pred = model.predict(self.X[self.test_idxs])
        y_true = self.y[self.test_idxs]

        scores = {}
        for metric in self.metrics:
            scores[metric.__name__] = metric(y_true, y_pred, multioutput=multioutput)

        # make sure scores are lists/arrays
        scores = {
            k: [v] if not isinstance(v, (list, np.ndarray)) else v
            for k, v in scores.items()
        }

        scores_df = (
            pd.DataFrame(scores)
            .assign(
                target=[f"target_{i}" for i in range(len(scores[next(iter(scores))]))]
            )
            .assign(short=get_short_model_name(model))
            .assign(model=get_model_name(model))
            .reindex(columns=["model", "short", "target"] + list(scores.keys()))
        ).round(4)

        # if multioutput is not raw_values, drop the target column
        if multioutput != "raw_values":
            scores_df = scores_df.drop(columns=["target"])

        return scores_df

    def plot_eval(
        self,
        model,
        style="Xy",
        n_cols=3,
        figsize=None,
        output_index=0,
        input_index=0,
    ):
        """Visualise different model evaluations on the test set.

        Parameters
        ----------
        model : object
            Fitted model.
        plot_type : str, optional
            The type of plot to draw:
            "Xy" observed and predicted values vs. features, including 2σ error bands where available (default).
            "actual_vs_predicted" draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
            "residual_vs_predicted" draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).
        n_cols : int, optional
            Number of columns in the plot grid for multi-output. Default is 2.
        output_index : int
            Index of the output to plot. Default is 0..
        input_index : int
            Index of the input to plot. Default is 0. Only used if plot_type="Xy".
        """
        fig = _plot_model(
            model,
            self.X[self.test_idxs],
            self.y[self.test_idxs],
            style,
            n_cols,
            figsize,
            input_index=input_index,
            output_index=output_index,
        )

        return fig

    def sensitivity_analysis(self, model, problem, N=1000, plot=True):
        """
        Perform sensitivity analysis on a fitted emulator.

        Parameters:
        -----------
        model_name : str
            The name of the fitted model to analyze.
        problem : dict
            The problem definition, including 'num_vars', 'names', and 'bounds'.
        N : int, optional
            The number of samples to generate (default is 1000).
        plot : bool, optional
            Whether to plot the results (default is True).

        Returns:
        --------
        dict
            A dictionary containing the Sobol indices.
        """
        Si = perform_sobol_analysis(model, problem, N)

        if plot:
            plot_sensitivity_indices(Si, problem)

        return Si
