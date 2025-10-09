import copy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y
from tqdm.auto import tqdm

from autoemulate.cross_validate import _run_cv
from autoemulate.cross_validate import _sum_cv
from autoemulate.cross_validate import _sum_cvs
from autoemulate.data_splitting import _split_data
from autoemulate.emulators import model_registry
from autoemulate.hyperparam_searching import _optimize_params
from autoemulate.logging_config import _configure_logging
from autoemulate.metrics import METRIC_REGISTRY
from autoemulate.model_processing import AutoEmulatePipeline
from autoemulate.plotting import _plot_cv
from autoemulate.plotting import _plot_model
from autoemulate.preprocess_target import get_dim_reducer
from autoemulate.preprocess_target import NonTrainableTransformer
from autoemulate.printing import _print_setup
from autoemulate.save import ModelSerialiser
from autoemulate.sensitivity_analysis import _plot_morris_analysis
from autoemulate.sensitivity_analysis import _plot_sobol_analysis
from autoemulate.sensitivity_analysis import _sensitivity_analysis
from autoemulate.utils import _check_cv
from autoemulate.utils import _ensure_2d
from autoemulate.utils import _get_full_model_name
from autoemulate.utils import _redirect_warnings
from autoemulate.utils import get_model_name
from autoemulate.utils import get_short_model_name


class AutoEmulate:
    """
    The AutoEmulate class is the main class of the AutoEmulate package. It is used to set up and compare
    different emulator models on a given dataset. It can also be used to summarise and visualise results,
    to save and load models and to run sensitivity analysis.
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
        scale_output=True,
        scaler_output=StandardScaler(),
        reduce_dim_output=False,
        preprocessing_methods=None,
        cross_validator=KFold(
            n_splits=5, shuffle=True, random_state=np.random.randint(1e5)
        ),
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
            Type of hyperparameter search to perform. Currently only "random", which picks random parameter settings
            from a grid param_search_iters times.
        param_search_iters : int
            Number of parameter settings that are sampled. Only used if
            param_search=True.
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
        scale_output : bool
            Whether to scale the output data.
        scaler_output : sklearn.preprocessing.StandardScaler
            Scaler to use. Defaults to StandardScaler. Can be any sklearn scaler.
        reduce_dim_output : bool
            Whether to reduce the dimensionality of the output data.
        preprocessing_methods : dict
            List of dictionaries with preprocessing methods and their parameters.
            Dimensionality reduction method to use for outputs. Can be PCA or Variational Autoencoder. #TODO: set PCA as default.
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

        if preprocessing_methods is None or len(preprocessing_methods) == 0:
            preprocessing_methods = [{"name": "None", "params": {}}]
        self.preprocessing_methods = preprocessing_methods

        # Replaced `self.models = _process_models(...)` with:
        self.ae_pipeline = AutoEmulatePipeline(
            model_registry=self.model_registry,
            model_names=list(self.model_names.keys()),
            y=self.y,
            prep_config={"name": "None", "params": {}},
            scale_input=scale,
            scaler_input=scaler,
            reduce_dim_input=reduce_dim,
            dim_reducer_input=dim_reducer,
            scale_output=scale_output,
            scaler_output=scaler_output,
            reduce_dim_output=reduce_dim_output,
        )
        # to avoid confusion between the pipeline and the actual models (which are now components of the pipeline).
        # The `AutoEmulatePipeline` class was created to wrap the pipeline creation logic and provide direct access
        # to its components (e.g., models, reducers, scalers, etc.).

        self.metrics = self._get_metrics(METRIC_REGISTRY)
        self.cross_validator = _check_cv(cross_validator)
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
        self.scale_output = scale_output
        self.scaler_output = scaler_output
        self.reduce_dim_output = reduce_dim_output
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
        """Compares models using cross-validation, with the option
        to perform hyperparameter search. self.setup() must be run first.

        Returns
        -------
        self.best_model : object
            Emulator with the highest cross-validation R2 score.
        """
        if not self.is_set_up:
            raise RuntimeError("Must run setup() before compare()")

        # Freshly initialise scores dataframe when running compare()
        self.scores_df = pd.DataFrame(
            columns=["model", "short", "preprocessing", "metric", "fold", "score"]
        ).astype(
            {
                "model": "object",
                "short": "object",
                "preprocessing": "object",
                "metric": "object",
                "fold": "int64",
                "score": "float64",
            }
        )

        if self.param_search:
            pb_text = "Optimising and cross-validating"
        else:
            pb_text = "Cross-validating"

        # Initialize dictionary to store results
        self.preprocessing_results = {}

        with tqdm(
            total=len(self.ae_pipeline.models_piped) * len(self.preprocessing_methods),
            desc=pb_text,
        ) as pbar:
            for prep_config in self.preprocessing_methods:
                # Create the actual transformer instance and fit it
                prep_name = prep_config["name"]
                prep_params = prep_config.get("params", {})

                # Outer loop training:
                # Fit the scaler and reducer on the whole training set
                if self.scale_output:
                    fitted_scaler = self.scaler_output.fit(_ensure_2d(self.y))
                    self.ae_pipeline.scaler_output = NonTrainableTransformer(
                        fitted_scaler
                    )
                if self.reduce_dim_output:
                    if self.scale_output:
                        # Fit the dimensionality reducer on the scaled output
                        fitted_reducer = get_dim_reducer(prep_name, **prep_params).fit(
                            fitted_scaler.transform(self.y)
                        )
                    else:
                        # Fit the dimensionality reducer on the original output
                        fitted_reducer = get_dim_reducer(prep_name, **prep_params).fit(
                            self.y
                        )
                    self.ae_pipeline.dim_reducer_output = NonTrainableTransformer(
                        fitted_reducer
                    )

                if self.scale_output or self.reduce_dim_output:
                    self.ae_pipeline._wrap_model_reducer_in_pipeline()

                # Initialize storage for this preprocessing method
                self.preprocessing_results[prep_name] = {
                    "models": self.ae_pipeline.models_piped,
                    "cv_results": {},
                    "best_model": None,
                    "transformer": self.ae_pipeline.dim_reducer_output,
                    "params": prep_params,
                }

                # Process each model
                for i, model in enumerate(
                    self.preprocessing_results[prep_name]["models"]
                ):
                    model_name = get_model_name(model)

                    # Update progress bar description
                    if prep_name == "None":
                        pbar.set_description(f"{pb_text} {model_name}")
                    else:
                        pbar.set_description(
                            f"{pb_text} {prep_name} | Model: {model_name}"
                        )

                    with _redirect_warnings(self.logger):
                        try:
                            # hyperparameter search
                            if self.param_search:
                                model = _optimize_params(
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
                            error_msg = f"Error cross-validating {model_name}"
                            if prep_name != "None":
                                error_msg = (
                                    f"Error cross-validating {prep_name} - {model_name}"
                                )
                            self.logger.exception(error_msg)
                            pbar.update(1)
                            continue

                        # Store the fitted model and results
                        self.preprocessing_results[prep_name]["models"][
                            i
                        ] = fitted_model
                        self.preprocessing_results[prep_name]["cv_results"][
                            model_name
                        ] = cv_results

                        # Add results to scores_df with preprocessing method
                        for metric_name, metric_values in cv_results.items():
                            if metric_name.startswith("test_"):
                                metric = metric_name.replace("test_", "")
                                for fold_idx, fold_score in enumerate(metric_values):
                                    new_row = {
                                        "model": model_name,
                                        "short": get_short_model_name(model),
                                        "preprocessing": prep_name,
                                        "metric": metric,
                                        "fold": fold_idx,
                                        "score": fold_score,
                                    }
                                    self.scores_df = pd.concat(
                                        [self.scores_df, pd.DataFrame([new_row])],
                                        ignore_index=True,
                                    )

                        # Update progress bar
                        pbar.update(1)

                # Get best model for this preprocessing method
                self.preprocessing_results[prep_name][
                    "best_model"
                ] = self.get_best_model_for_prep(
                    prep_results=self.preprocessing_results[prep_name], metric="r2"
                )

        # Find the overall best model and preprocessing method
        (
            self.best_prep_method,
            self.best_model,
            self.best_transformer,
        ) = self.get_overall_best_model(metric="r2")

        # Store additional information about the best combination
        self.best_combination = {
            "preprocessing": self.best_prep_method,
            "model": get_model_name(self.best_model),
            "transformer": self.best_transformer.base_transformer_name,
        }

        return self.best_model

    def get_model(self, name=None, rank=None, preprocessing=None, metric="r2"):
        """Get a fitted model by name or rank, optionally from specific preprocessing.

        Parameters
        ----------
        name : str, optional
            Model name to get (e.g., "GaussianProcess")
        rank : int, optional
            Rank of model to get (1 = best)
        preprocessing : str, optional
            Specific preprocessing method to use (None for best overall)
        metric : str, optional
            Metric to use for ranking ("r2" by default)

        Returns
        -------
        Fitted model instance
        """
        if not hasattr(self, "preprocessing_results"):
            raise RuntimeError("Must run compare() first")

        # Handle rank specification
        if rank is not None:
            if preprocessing is None:
                # Get overall ranking across all preprocessing methods
                summary = self.summarise_cv(sort_by=metric)
                # Remove duplicate models from different preprocessing
                summary = summary.drop_duplicates(
                    subset=["model", "short"], keep="first"
                )
                if rank < 1 or rank > len(summary):
                    raise RuntimeError(f"Rank must be between 1 and {len(summary)}")
                name = summary.iloc[rank - 1]["model"]
            else:
                # Get ranking within specific preprocessing method
                if preprocessing not in self.preprocessing_results:
                    raise ValueError(
                        f"Unknown preprocessing: {preprocessing}. "
                        f"Available: {list(self.preprocessing_results.keys())}"
                    )
                summary = self.summarise_cv(preprocessing=preprocessing, sort_by=metric)
                if rank < 1 or rank > len(summary):
                    raise RuntimeError(
                        f"Rank must be between 1 and {len(summary)} for preprocessing '{preprocessing}'"
                    )
                name = summary.iloc[rank - 1]["model"]

        # Get model by name with optional preprocessing filter
        if name is not None:
            # Convert short name to full name if needed
            full_name = _get_full_model_name(name, self.model_names)
            if full_name is None:
                raise ValueError(f"Model '{name}' not found in registered models")

            if preprocessing is not None:
                # Get model from specific preprocessing method
                if preprocessing not in self.preprocessing_results:
                    raise ValueError(
                        f"Unknown preprocessing: {preprocessing}. "
                        f"Available: {list(self.preprocessing_results.keys())}"
                    )
                for model in self.preprocessing_results[preprocessing]["models"]:
                    if get_model_name(model) == full_name:
                        return model
                raise ValueError(
                    f"Model '{full_name}' not found in preprocessing '{preprocessing}'"
                )
            else:
                # Search all preprocessing methods for the best instance of this model
                best_score = -float("inf") if metric == "r2" else float("inf")
                best_model = None

                for prep_name, prep_data in self.preprocessing_results.items():
                    # Get CV results for this preprocessing method
                    cv_results = prep_data["cv_results"]
                    if full_name not in cv_results:
                        continue

                    # Get mean score for this model+preprocessing combination
                    means = _sum_cvs({full_name: cv_results[full_name]}, metric)
                    current_score = means.iloc[0][f"{metric}"]

                    # Find the model with the correct name that has the best score
                    for model in prep_data["models"]:
                        if get_model_name(model) == full_name:
                            if (metric == "r2" and current_score > best_score) or (
                                metric != "r2" and current_score < best_score
                            ):
                                best_score = current_score
                                best_model = model
                            break

                if best_model is None:
                    raise ValueError(
                        f"Model '{full_name}' not found in any preprocessing method"
                    )
                return best_model

        # Get best model for specific preprocessing or overall best
        if preprocessing is not None:
            if preprocessing not in self.preprocessing_results:
                raise ValueError(
                    f"Unknown preprocessing: {preprocessing}. "
                    f"Available: {list(self.preprocessing_results.keys())}"
                )
            return self.get_best_model_for_prep(
                prep_results=self.preprocessing_results[preprocessing], metric=metric
            )
        else:
            # Get overall best model across all preprocessing methods
            _, best_model, _ = self.get_overall_best_model(metric=metric)
            return best_model

    def get_best_model_for_prep(self, prep_results, metric="r2"):
        """Get the best model for a specific preprocessing method.

        Parameters
        ----------
        prep_results : dict
            The preprocessing results dictionary containing 'cv_results'
        metric : str
            The metric to use for comparison (default: 'r2')

        Returns
        -------
        object
            The best model for this preprocessing method
        """
        cv_results = prep_results["cv_results"]

        means = _sum_cvs(cv_results, metric)
        best_model_name = means.iloc[0]["model"]

        for model in prep_results["models"]:
            if get_model_name(model) == best_model_name:
                return model

    def get_overall_best_model(self, metric="r2"):
        """Get the best model across all preprocessing methods.

        Parameters
        ----------
        metric : str
            The metric to use for comparison (default: 'r2')

        Returns
        -------
        tuple
            (best_preprocessing_method, best_model, best_transformer)
        """
        best_score = -float("inf") if metric == "r2" else float("inf")
        best_prep = None
        best_model = None
        best_transformer = None

        for prep_name, prep_data in self.preprocessing_results.items():
            # Get mean scores for this preprocessing method
            cv_results = prep_data["cv_results"]
            means = _sum_cvs(cv_results, metric)
            # Get the best score for this preprocessing
            top_score = means.iloc[0][f"{metric}"]
            # Compare with current best
            if (metric == "r2" and top_score > best_score) or (
                metric != "r2" and top_score < best_score
            ):
                best_score = top_score
                best_prep = prep_name
                best_model = prep_data["best_model"]
                best_transformer = prep_data["transformer"]

        if best_model is None:
            raise RuntimeError("No valid models found for comparison")

        return best_prep, best_model, best_transformer

    def refit(self, model=None):
        """Refits model on full data. This is useful, as `compare()` runs only on the training data.
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
        """Print the parameters of the AutoEmulate object."""
        _print_setup(self)

    def summarise_cv(self, model=None, preprocessing=None, sort_by="r2"):
        """Summarise cross-validation results across models and preprocessing methods.

        Parameters
        ----------
        model : str, optional
            Name of the model to get cv results for (can be full name like "GaussianProcess"
            or short name like "gp"). If None, summarises results for all models.
        preprocessing : str, optional
            Name of preprocessing method to filter by. If None, includes all methods.
        sort_by : str, optional
            The metric to sort by. Default is "r2", can also be "rmse".

        Returns
        -------
        pandas.DataFrame
            DataFrame summarising cv results with preprocessing information.
        """
        if not hasattr(self, "preprocessing_results"):
            raise RuntimeError("Must run compare() before summarise_cv()")

        # Convert model short name to full name if needed
        model_name = None
        if model is not None:
            model_name = _get_full_model_name(model, self.model_names)
            if model_name is None:
                raise ValueError(f"Model '{model}' not found in registered models")

        all_results = []

        for prep_name, prep_data in self.preprocessing_results.items():
            # Skip if preprocessing filter doesn't match
            if preprocessing is not None and prep_name != preprocessing:
                continue

            # Filter models if requested
            for m_name, cv_res in prep_data["cv_results"].items():
                if model_name is not None and m_name != model_name:
                    continue

                # Get the actual model object to determine short name
                model_obj = next(
                    (m for m in prep_data["models"] if get_model_name(m) == m_name),
                    None,
                )
                if model_obj is None:
                    continue

                # Get summary for this model+preprocessing combination
                model_summary = _sum_cv(cv_res)
                model_summary["preprocessing"] = prep_name
                model_summary["model"] = m_name
                model_summary["short"] = get_short_model_name(model_obj)
                all_results.append(model_summary)

        if not all_results:
            raise ValueError("No results found for the specified filters")

        # Combine and sort results
        full_results = pd.concat(all_results)
        column_order = ["preprocessing", "model", "short"] + [
            c
            for c in full_results.columns
            if c not in ["preprocessing", "model", "short"]
        ]
        full_results = full_results[column_order]

        # Determine sort column and direction
        sort_column = (
            f"mean_{sort_by}" if f"mean_{sort_by}" in full_results.columns else sort_by
        )
        ascending = sort_by.lower() != "r2"

        return full_results.sort_values(
            by=["preprocessing", sort_column], ascending=[True, ascending]
        ).reset_index(drop=True)

    summarize_cv = summarise_cv  # alias

    def plot_cv(
        self,
        model=None,
        preprocessing=None,
        style="Xy",
        n_cols=3,
        figsize=None,
        output_index=0,
        input_index=0,
    ):
        """Plots the results of the cross-validation for a specific preprocessing method.

        Parameters
        ----------
        model : str
            Name of the model to plot. If None, plots best folds of each model.
        preprocessing : str
            Name of preprocessing method to plot. If None, uses the best preprocessing method.
            Use 'None' (string) for no preprocessing.
        style : str, optional
            The type of plot to draw.
        n_cols : int
            Maximum number of columns in the plot grid.
        figsize : tuple, optional
            Overrides the default figure size.
        output_index : int
            Index of the output to plot.
        input_index : int
            Index of the input to plot.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the plot.
        """
        if not hasattr(self, "preprocessing_results"):
            raise RuntimeError("Must run compare() before plot_cv()")

        # Handle preprocessing selection
        if preprocessing is None:
            preprocessing = self.best_prep_method
            print(f"Using best preprocessing method: {preprocessing}")
        elif preprocessing not in self.preprocessing_results:
            raise ValueError(
                f"Unknown preprocessing method: {preprocessing}. "
                f"Available methods: {list(self.preprocessing_results.keys())}"
            )

        # Print preprocessing information
        if preprocessing == "None":
            print("\nNo preprocessing was applied (using raw target values)")
        else:
            prep_config = next(
                m for m in self.preprocessing_methods if m["name"] == preprocessing
            )
            print("\nPreprocessing Configuration:")
            print(f"Method: {preprocessing}")
            print("Hyperparameters:")
            for param, value in prep_config["params"].items():
                print(f"  {param}: {value}")

        # Get model name if specified
        model_name = (
            _get_full_model_name(model, self.model_names) if model is not None else None
        )

        # Get the appropriate CV results
        if model_name:
            cv_results = self.preprocessing_results[preprocessing]["cv_results"].get(
                model_name
            )
            if cv_results is None:
                raise ValueError(
                    f"Model {model_name} not found for preprocessing {preprocessing}"
                )
        else:
            cv_results = {}
            for m_name, m_results in self.preprocessing_results[preprocessing][
                "cv_results"
            ].items():
                cv_results[m_name] = m_results

        # Use original or transformed y values
        y_train = self.y[self.train_idxs]
        if preprocessing != "None":
            transformer = self.preprocessing_results[preprocessing]["transformer"]
            y_train = transformer.transform(y_train)

        # Create the plot
        figure = _plot_cv(
            self.preprocessing_results[preprocessing][
                "cv_results"
            ],  # TODO: debug why this and not directly cv_results
            self.X[self.train_idxs],
            y_train,
            model_name=model_name,
            n_cols=n_cols,
            style=style,
            figsize=figsize,
            output_index=output_index,
            input_index=input_index,
        )

        # Add preprocessing info to title
        prep_title = (
            "No preprocessing"
            if preprocessing == "None"
            else f"Preprocessing: {preprocessing}"
        )
        if figure._suptitle is not None:
            figure._suptitle.set_text(figure._suptitle.get_text() + f" | {prep_title}")
        else:
            figure.suptitle(prep_title)

        return figure

    def evaluate(self, model=None, preprocessing=None, multioutput="uniform_average"):
        """
        Evaluates the model on the test set, handling preprocessing transformations if any.

        Parameters
        ----------
        model : object, optional
            Fitted model to evaluate. If None, uses the best model from comparison.
        preprocessing : str, optional
            Name of preprocessing method used for the model. If None, uses the best preprocessing method
            from comparison or assumes no preprocessing if none was specified.
        multioutput : str, optional
            Defines aggregating of multiple output scores:
            - 'raw_values' : returns scores for each target
            - 'uniform_average' : scores are averaged uniformly
            - 'variance_weighted' : scores are averaged weighted by their individual variances

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the model scores on the test set with columns:
            - model: Model name
            - short: Short model name
            - preprocessing: Preprocessing method used (if any)
            - target: Target/output name (if multioutput='raw_values')
            - metric scores (e.g., r2, rmse)
        """
        # select model and preprocessing method and transformer if available
        if model is None:
            model = self.best_model

        if preprocessing is None:
            if hasattr(self, "best_prep_method"):
                preprocessing = self.best_prep_method
            else:
                preprocessing = "None"

        transformer = None
        if (
            hasattr(self, "preprocessing_results")
            and preprocessing in self.preprocessing_results
        ):
            transformer = self.preprocessing_results[preprocessing]["transformer"]

        # Get true values (transform if needed)
        # y_true = self.y[self.test_idxs]
        # if transformer is not None:
        #    _, y_true = transformer.transform(self.X[self.test_idxs], y_true)

        # Get predictions
        y_pred = model.predict(self.X[self.test_idxs])
        y_true = self.y[self.test_idxs]

        # If preprocessing was applied, inverse transform predictions for evaluation
        # if transformer is not None and hasattr(transformer, "inverse_transform"):
        #    y_pred = transformer.inverse_transform(self.X[self.test_idxs], y_pred)[1]
        #   y_true = self.y[self.test_idxs]  # Revert to original y values

        # Calculate metrics
        scores = {}
        for metric in self.metrics:
            scores[metric.__name__] = metric(y_true, y_pred, multioutput=multioutput)

        # Prepare results dataframe
        scores = {
            k: [v] if not isinstance(v, (list, np.ndarray)) else v
            for k, v in scores.items()
        }

        # Create base dataframe
        scores_df = pd.DataFrame(scores)

        # Add target column if multioutput is raw_values
        if multioutput == "raw_values":
            scores_df["target"] = [
                f"y{i}" for i in range(len(scores[next(iter(scores))]))
            ]

        # Add metadata columns
        scores_df["model"] = get_model_name(model)
        scores_df["short"] = get_short_model_name(model)
        scores_df["preprocessing"] = preprocessing

        # Reorder columns
        cols = ["model", "short", "preprocessing"]
        if multioutput == "raw_values":
            cols.append("target")
        cols.extend(list(scores.keys()))
        scores_df = scores_df[cols].round(4)

        return scores_df

    def plot_eval(
        self,
        model=None,
        preprocessing=None,
        style="Xy",
        n_cols=3,
        figsize=None,
        output_index=0,
        input_index=0,
    ):
        """Visualise model predictive performance on the test set.

        Parameters
        ----------
        model : object
            Fitted model.
        style : str, optional
            The type of plot to draw:
            "Xy" observed and predicted values vs. features, including 2σ error bands where available (default).
            "actual_vs_predicted" draws the observed values (y-axis) vs. the predicted values (x-axis) (default).
            "residual_vs_predicted" draws the residuals, i.e. difference between observed and predicted values, (y-axis) vs. the predicted values (x-axis).
        n_cols : int, optional
            Maximum number of columns in the plot grid for multi-output. Default is 3.
        output_index : list, int
            Index of the output to plot. Either a single index or a list of indices.
        input_index : list, int
            Index of the input to plot. Either a single index or a list of indices. Only used if style="Xy".
        """
        # select model and preprocessing method and transformer if available
        if model is None:
            model = self.best_model

        """
        transformer = None
        if (
            hasattr(self, "preprocessing_results")
            and preprocessing in self.preprocessing_results
        ):
            transformer = self.preprocessing_results[preprocessing]["transformer"]

        scaler_output = None
        if (
            hasattr(model.transformer.named_steps, "scaler_output")
            and preprocessing in self.preprocessing_results
        ):
            scaler_output = model.transformer.named_steps["scaler_output"]
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

    def sensitivity_analysis(
        self,
        model=None,
        method="sobol",
        problem=None,
        N=1024,
        conf_level=0.95,
    ):
        """Perform Sobol sensitivity analysis on a fitted emulator.

        Sobol sensitivity analysis is a variance-based method that decomposes the variance of the model
        output into contributions from individual input parameters and their interactions. It calculates:
        - First-order indices (S1): Direct contribution of each input parameter
        - Second-order indices (S2): Contribution from pairwise interactions between parameters
        - Total-order indices (ST): Total contribution of a parameter, including all its interactions

        Parameters
        ----------
        model : object, optional
            Fitted model. If None, uses the best model from cross-validation.
        problem : dict, optional
            The problem definition dictionary. If None, the problem is generated from X using
            minimum and maximum values of the features as bounds. The dictionary should contain:

            - 'num_vars': Number of input variables (int)
            - 'names': List of variable names (list of str)
            - 'bounds': List of [min, max] bounds for each variable (list of lists)
            - 'output_names': Optional list of output names (list of str)

            Example::

                problem = {
                    "num_vars": 2,
                    "names": ["x1", "x2"],
                    "bounds": [[0, 1], [0, 1]],
                    "output_names": ["y1", "y2"]  # optional
                }
        N : int, optional
            Number of samples to generate for the analysis. Higher values give more accurate
            results but increase computation time. Default is 1024.
        conf_level : float, optional
            Confidence level (between 0 and 1) for calculating confidence intervals of the
            sensitivity indices. Default is 0.95 (95% confidence).

        Returns
        -------
        pandas.DataFrame
            - 'parameter': Input parameter name
            - 'output': Output variable name
            - 'S1', 'S2', 'ST': First, second, and total order sensitivity indices
            - 'S1_conf', 'S2_conf', 'ST_conf': Confidence intervals for each index

        Notes
        -----
        The analysis requires N * (2D + 2) model evaluations, where D is the number of input
        parameters. For example, with N=1024 and 5 parameters, this requires 12,288 evaluations.
        """
        self.method = method
        if method not in ["sobol", "morris"]:
            raise ValueError(f"Unknown method: {method}. Must be 'sobol' or 'morris'.")

        if model is None:
            if not hasattr(self, "best_model"):
                raise RuntimeError("Must run compare() before sensitivity_analysis()")
            model = self.refit(self.best_model)
            self.logger.info(
                f"No model provided, using {get_model_name(model)}, which had the highest average cross-validation score, refitted on full data."
            )

        df_results = _sensitivity_analysis(
            model=model,
            method=method,
            problem=problem,
            X=self.X,
            N=N,
            conf_level=conf_level,
        )

        return df_results

    def plot_sensitivity_analysis(
        self, results, index="S1", param_groups=None, n_cols=None, figsize=None
    ):
        """
        Plot the sensitivity analysis results.

        Parameters:
        -----------
        results : pd.DataFrame
            The results from sobol_results_to_df.
        index : str, default "S1"
            The type of sensitivity index to plot.
            - "S1": first-order indices
            - "S2": second-order/interaction indices
            - "ST": total-order indices
        n_cols : int, optional
            The number of columns in the plot. Defaults to 3 if there are 3 or more outputs,
            otherwise the number of outputs.
        figsize : tuple, optional
            Figure size as (width, height) in inches.If None, automatically calculated.

        """
        if self.method == "sobol":
            return _plot_sobol_analysis(results, index, n_cols, figsize)
        elif self.method == "morris":
            return _plot_morris_analysis(results, param_groups, n_cols, figsize)
