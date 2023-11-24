import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.validation import check_X_y
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from autoemulate.metrics import METRIC_REGISTRY
from autoemulate.emulators import MODEL_REGISTRY
from autoemulate.cv import CV_REGISTRY
from autoemulate.logging_config import configure_logging
from autoemulate.plotting import plot_results
from autoemulate.hyperparam_search import HyperparamSearcher


class AutoEmulate:
    def __init__(self):
        """Initializes an AutoEmulate object."""
        self.X = None
        self.y = None
        self.is_set_up = False

    def setup(
        self,
        X,
        y,
        use_grid_search=False,
        grid_search_type="random",
        grid_search_iters=20,
        normalise=True,
        scaler=StandardScaler(),
        fold_strategy="kfold",
        folds=5,
        n_jobs=None,
        log_to_file=False,
    ):
        """Sets up the automatic emulation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Simulation input.
        y : array-like, shape (n_samples, n_outputs)
            Simulation output.
        use_grid_search : bool
            Whether to perform hyperparameter search over predifined parameter grids.
        grid_search_type : str
            Type of hyperparameter search to perform. Can be "grid", "random", or "bayes".
        grid_search_iters : int
            Number of parameter settings that are sampled. Only used if
            use_grid_search=True and grid_search_type="random".
        normalise : bool, default=False
            Whether to normalise the data before fitting the models. Currently only
            z-transformation using StandardScaler.
        fold_strategy : str
            Cross-validation strategy, currently either "kfold" or "stratified_kfold".
        folds : int
            Number of folds.
        n_jobs : int
            Number of jobs to run in parallel. `None` means 1, `-1` means using all processors.
        log_to_file : bool
            Whether to log to file.
        """
        self.X, self.y = self._check_input(X, y)
        self.models = self._wrap_models_in_pipeline(
            self._get_models(MODEL_REGISTRY), normalise=normalise, scaler=scaler
        )
        self.metrics = self._get_metrics(METRIC_REGISTRY)
        self.cv = CV_REGISTRY[fold_strategy](folds=folds, shuffle=True)
        self.use_grid_search = use_grid_search
        self.search_type = grid_search_type
        self.grid_search_iters = grid_search_iters
        self.normalise = normalise
        self.scaler = scaler
        self.n_jobs = n_jobs
        self.logger = configure_logging(log_to_file=log_to_file)
        self.is_set_up = True
        self.cv_results = {}

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

    def _get_models(self, MODEL_REGISTRY):
        """Get models from REGISTRY

        Parameters
        ----------
        MODEL_REGISTRY : dict
            Registry of models.

        Returns
        -------
        list
            List of models.
        """
        return [model() for model in MODEL_REGISTRY.values()]

    def _wrap_models_in_pipeline(self, models, normalise, scaler):
        """Create pipelines from models

        Parameters
        ----------
        models : list
            List of models.
        normalise : bool
            If True, add scaler to models.
        scaler : sklearn.preprocessing.StandardScaler
            Scaler to use. Defaults to StandardScaler.

        Returns
        -------
        list
            List of models wrapped in pipelines.
        """
        if normalise:
            models = [
                Pipeline([("scaler", scaler), ("model", model)]) for model in models
            ]
        else:
            models = [Pipeline([("model", model)]) for model in models]
        return models

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
            columns=["model", "metric", "fold", "score"]
        ).astype(
            {"model": "object", "metric": "object", "fold": "int64", "score": "float64"}
        )
        # Freshly initialise best parameters for each model
        self.best_params = {}

        for i in range(len(self.models)):
            if self.use_grid_search:
                self.models[i] = self._update_to_best_hyperparams(
                    self.models[i], self.search_type
                )
            self._cross_validate(self.models[i])

        # returns best model fitted on full data
        return self._get_best_model(metric="r2")

    def _cross_validate(self, model):
        """Perform cross-validation on a given model using the specified metrics.

        Parameters
        ----------
            model: A scikit-learn estimator object.

        Class attributes used
        ---------------------
            self.X : array-like, shape (n_samples, n_features)
                Simulation input.
            self.y : array-like, shape (n_samples, n_outputs)
                Simulation output.
            self.cv : scikit-learn cross-validation object
                Cross-validation strategy.
            self.metrics : list of str
                List of metrics to use for cross-validation.
            self.n_jobs : int
                Number of jobs to run in parallel. `None` means 1, `-1` means using all processors.

        Returns
        -------
            scores_df : pandas.DataFrame
                Updates dataframe containing the cv scores the model.

        """

        # Get model name
        model_name = type(model.named_steps["model"]).__name__

        # The metrics we want to use for cross-validation
        scorers = {metric.__name__: make_scorer(metric) for metric in self.metrics}

        self.logger.info(f"Cross-validating {model_name}...")
        self.logger.info(f"Parameters: {model.named_steps['model'].get_params()}")

        # Cross-validate
        cv_results = cross_validate(
            model,
            self.X,
            self.y,
            cv=self.cv,
            scoring=scorers,
            n_jobs=self.n_jobs,
            return_estimator=True,
            return_indices=True,
        )
        # updates pandas dataframe with model cv scores
        self._update_scores_df(model_name, cv_results)
        # save results for plotting etc.
        self.cv_results[model_name] = cv_results

    def _update_scores_df(self, model_name, cv_results):
        """Updates the scores dataframe with the results of the cross-validation.

        Parameters
        ----------
            model_name : str
                Name of the model.
            cv_results : dict
                Results of the cross-validation.

        Returns
        -------
            None
                Modifies the self.scores_df DataFrame in-place.

        """
        # Gather scores from each metric
        # Initialise scores dataframe
        for key in cv_results.keys():
            if key.startswith("test_"):
                for fold, score in enumerate(cv_results[key]):
                    self.scores_df.loc[len(self.scores_df.index)] = {
                        "model": model_name,
                        "metric": key.split("test_", 1)[1],
                        "fold": fold,
                        "score": score,
                    }

    def _update_to_best_hyperparams(self, model, search_type):
        """Performs hyperparameter search and updates the model.

        Parameters
        ----------
        model_index : int
            Index of the model in self.models.
        model : scikit-learn estimator object
            Model to perform hyperparameter search on.
        """
        # Perform hyperparameter search and update model
        hyperparam_searcher = HyperparamSearcher(
            X=self.X,
            y=self.y,
            cv=self.cv,
            n_jobs=self.n_jobs,
            logger=self.logger,
        )
        best_params = hyperparam_searcher.search(
            model,
            search_type=search_type,
            param_grid=None,
            niter=self.grid_search_iters,
        )
        # Update model with best parameters
        model.set_params(**best_params)
        # Update best parameter list
        model_name = type(model.named_steps["model"]).__name__
        self.best_params[model_name] = best_params

        return model

    def _get_best_model(self, metric="r2"):
        """Determine the best model using average cv score

        Parameters
        ----------
        metric : str
            Metric to use for determining the best model.

        Returns
        -------
        best_model : object
            Best model fitted on full data. If normalised, returns a pipeline with
            the scaler and the best model.
        """

        # best model name
        means = (
            self.scores_df.groupby(["model", "metric"])["score"]
            .mean()
            .unstack()
            .reset_index()
        )
        # get best model name
        best_model_name = means.loc[means[metric].idxmax(), "model"]

        # get best model:
        for model in self.models:
            if type(model.named_steps["model"]).__name__ == best_model_name:
                best_model = model
                break

        self.logger.info(
            f"{best_model_name} is the best model, refitting on full dataset..."
        )
        # refit best model on full dataset
        best_model.fit(self.X, self.y)

        return best_model

    def print_results(self, model=None):
        # check if model is in self.models
        if model is not None:
            model_names = [type(model.steps[-1][1]).__name__ for model in self.models]
            if model not in model_names:
                raise ValueError(
                    f"Model {model} not found. Available models are: {model_names}"
                )
        if model is None:
            means = (
                self.scores_df.groupby(["model", "metric"])["score"]
                .mean()
                .unstack()
                .reset_index()
            )
            print("Average scores across all models:")
            print(means)
        else:
            scores = self.scores_df[self.scores_df["model"] == model].pivot(
                index="fold", columns="metric", values="score"
            )
            print(f"Scores for {model} across all folds:")
            print(scores)

    def plot_results(self, model_name=None):
        """Plots the results of the cross-validation.

        Parameters
        ----------
        model_name : str
            Name of the model to plot. If None, plots best folds of each models.
            If a model name is specified, plots all folds of that model.
        """
        plot_results(self.cv_results, self.X, self.y, model_name=model_name)
