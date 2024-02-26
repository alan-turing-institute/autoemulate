from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y

from autoemulate.cross_validate import run_cv
from autoemulate.cross_validate import update_scores_df
from autoemulate.cv import CV_REGISTRY
from autoemulate.data_splitting import split_data
from autoemulate.emulators import MODEL_REGISTRY
from autoemulate.hyperparam_searching import optimize_params
from autoemulate.logging_config import configure_logging
from autoemulate.metrics import METRIC_REGISTRY
from autoemulate.model_processing import get_and_process_models
from autoemulate.plotting import _plot_model
from autoemulate.plotting import _plot_results
from autoemulate.printing import _print_cv_results
from autoemulate.save import ModelSerialiser
from autoemulate.utils import get_mean_scores
from autoemulate.utils import get_model_name


class AutoEmulate:
    def __init__(self):
        """Initializes an AutoEmulate object."""
        self.X = None
        self.y = None
        self.is_set_up = False

    def setup(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_search: bool = False,
        param_search_type: str = "random",
        param_search_iters: int = 20,
        test_set_size: float = 0.2,
        scale: bool = True,
        scaler: StandardScaler = StandardScaler(),
        reduce_dim: bool = False,
        dim_reducer: PCA = PCA(),
        fold_strategy: str = "kfold",
        folds: int = 5,
        n_jobs: Optional[int] = None,
        model_subset: Optional[list] = None,
        log_to_file: bool = False,
    ) -> None:
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
        fold_strategy : str
            Cross-validation strategy, currently either "kfold" or "stratified_kfold".
        folds : int
            Number of folds.
        n_jobs : int
            Number of jobs to run in parallel. `None` means 1, `-1` means using all processors.
        model_subset : list
            List of models to use. If None, uses all models in MODEL_REGISTRY.
            Currently, any of: SecondOrderPolynomial, RBF, RandomForest, GradientBoosting,
            GaussianProcessSk, SupportVectorMachines, XGBoost
        log_to_file : bool
            Whether to log to file.
        """
        self.X, self.y = self._check_input(X, y)
        self.train_idxs, self.test_idxs = split_data(
            self.X, test_size=test_set_size, random_state=42
        )
        self.models = get_and_process_models(
            MODEL_REGISTRY,
            model_subset,
            self.y,
            scale,
            scaler,
            reduce_dim,
            dim_reducer,
        )
        self.metrics = self._get_metrics(METRIC_REGISTRY)
        self.cv = self._get_cv(CV_REGISTRY, fold_strategy, folds)
        self.param_search = param_search
        self.search_type = param_search_type
        self.param_search_iters = param_search_iters
        self.scale = scale
        self.scaler = scaler
        self.n_jobs = n_jobs
        self.logger = configure_logging(log_to_file=log_to_file)
        self.is_set_up = True
        self.cv_results = {}

    def _check_input(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def _get_metrics(self, METRIC_REGISTRY: dict) -> list:
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

    def _get_cv(self, CV_REGISTRY: dict, fold_strategy: str, folds: int) -> KFold:
        """Get cross-validation strategy from REGISTRY

        Parameters
        ----------
        CV_REGISTRY : dict
            Registry of cross-validation strategies.
        fold_strategy : str
            Name of the cross-validation strategy. Currently only "kfold" is supported.
        folds : int
            Number of folds.

        Returns
        -------
        cv : sklearn.model_selection.KFold
            An instance of the KFold class.
        """
        return CV_REGISTRY[fold_strategy](folds=folds, shuffle=True)

    def compare(self) -> pd.DataFrame:
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
            columns=["model", "metric", "fold", "score"]
        ).astype(
            {
                "model": "object",
                "metric": "object",
                "fold": "int64",
                "score": "float64",
            }
        )

        for i in range(len(self.models)):
            try:
                # hyperparameter search
                if self.param_search:
                    self.models[i] = optimize_params(
                        X=self.X[self.train_idxs],
                        y=self.y[self.train_idxs],
                        cv=self.cv,
                        model=self.models[i],
                        search_type=self.search_type,
                        niter=self.param_search_iters,
                        param_space=None,
                        n_jobs=self.n_jobs,
                        logger=self.logger,
                    )

                # run cross validation
                fitted_model, cv_results = run_cv(
                    X=self.X[self.train_idxs],
                    y=self.y[self.train_idxs],
                    cv=self.cv,
                    model=self.models[i],
                    metrics=self.metrics,
                    n_jobs=self.n_jobs,
                    logger=self.logger,
                )
            except Exception as e:
                print(f"Error fitting model {get_model_name(self.models[i])}")
                print(e)  # should be replaced with logging
                continue

            self.models[i] = fitted_model
            self.cv_results[get_model_name(self.models[i])] = cv_results

            # update scores dataframe
            self.scores_df = update_scores_df(
                self.scores_df,
                self.models[i],
                self.cv_results[get_model_name(self.models[i])],
            )

        # returns best model fitted on full data
        self.best_model = self.get_model(rank=1, metric="r2")

        # print best model
        best_model_name = get_model_name(self.best_model)
        mean_scores = get_mean_scores(self.scores_df, "r2")
        self.logger.info(
            f"{best_model_name} is the best model with R^2 = {mean_scores.loc[mean_scores['model']==best_model_name, 'r2'].item():.3f}"
        )

        return self.best_model

    def get_model(self, rank: int = 1, metric: str = "r2"):  # TODO: add return type
        """Get a fitted model based on it's rank in the comparison.

        Parameters
        ----------
        rank : int
            Rank of the model to return. Defaults to 1, which is the best model, 2 is the second best, etc.
        metric : str
            Metric to use for determining the best model.

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

        return chosen_model

    def refit_model(self, model):  # TODO: add model type
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

    def save_model(
        self, model=None, filepath: str = None
    ) -> None:  # TODO add model type
        """Saves the best model to disk.

        Parameters
        ----------
        model : object
            Fitted model.
        filepath : str
            Path to the model file.

        Returns
        -------
        None
        """
        if not hasattr(self, "best_model"):
            raise RuntimeError("Must run compare() before save_model()")
        serialiser = ModelSerialiser()

        if model is None:
            model = self.best_model
        if filepath is None:
            raise ValueError("Filepath must be provided")

        serialiser.save_model(model, filepath)

    def load_model(self, filepath: str = None):  # TODO add return type (model type)
        """Loads a model from disk.

        Parameters
        ----------
        filepath : str
            Path to the model file.

        Returns
        -------
        model : object
            Loaded model.
        """
        serialiser = ModelSerialiser()
        if filepath is None:
            raise ValueError("Filepath must be provided")

        return serialiser.load_model(filepath)

    # TODO for print_results: suggestion, rename model to model_name here to not confuse with other references to the model object
    def print_results(self, model: Optional[str] = None, sort_by: str = "r2") -> None:
        """Print cv results.

        Parameters
        ----------
        model : str, optional
            The name of the model to print. If None, the best fold from each model will be printed.
            If a model name is provided, the scores for that model across all folds will be printed.
        sort_by : str, optional
            The metric to sort by. Default is "r2", can also be "rmse".
        """
        _print_cv_results(
            self.models,
            self.scores_df,
            model=model,
            sort_by=sort_by,
        )

    # TODO for plot_results: suggestion, rename model to model_name here to not confuse with other references to the model object
    def plot_results(
        self,
        model: Optional[str] = None,
        plot_type: str = "actual_vs_predicted",
        n_cols: int = 3,
        figsize: Optional[tuple] = None,
        output_index: int = 0,
    ):
        """Plots the results of the cross-validation.

        Parameters
        ----------
        model : str
            Name of the model to plot. If None, plots best folds of each models.
            If a model name is specified, plots all folds of that model.
        plot_type : str, optional
            The type of plot to draw:
            “actual_vs_predicted” draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
            “residual_vs_predicted” draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).
        n_cols : int
            Number of columns in the plot grid.
        figsize : tuple, optional
            Overrides the default figure size.
        output_index : int
            Index of the output to plot. Default is 0.
        """
        _plot_results(
            self.cv_results,
            self.X,
            self.y,
            model_name=model,
            n_cols=n_cols,
            plot_type=plot_type,
            figsize=figsize,
            output_index=output_index,
        )

    def evaluate_model(self, model=None) -> pd.DataFrame:  # TODO add model type
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
                pd.DataFrame(scores, index=[0]),
            ],
            axis=1,
        ).round(3)

        return scores_df

    def plot_model(
        self,
        model,
        plot: str = "standard",
        n_cols: int = 2,
        figsize: Optional[tuple] = None,
    ) -> None:  # TODO add model type
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
        figsize : tuple, optional
            Overrides the default figure size.
        """
        _plot_model(
            model,
            self.X[self.test_idxs],
            self.y[self.test_idxs],
            plot,
            n_cols,
            figsize,
        )
