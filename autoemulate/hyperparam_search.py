import logging

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV

from autoemulate.utils import adjust_param_grid
from autoemulate.utils import get_model_name
from autoemulate.utils import get_model_param_grid
from autoemulate.utils import get_model_params


class HyperparamSearcher:
    """Performs hyperparameter search for a given model."""

    def __init__(self, X, y, cv, n_jobs, logger=None):
        """Initializes a HyperparamSearch object.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        cv : int
            Number of folds in the cross-validation.
        n_jobs : int
            Number of jobs to run in parallel.
        niter : int, default=20
            Number of parameter settings that are sampled.
        logger : logging.Logger
            Logger object.
        """
        self.X = X
        self.y = y
        self.cv = cv
        self.n_jobs = n_jobs
        self.logger = logger if logger else logging.getLogger(__name__)

    def search(self, model, search_type="random", param_grid=None, niter=20):
        """Performs hyperparameter search for a given model.

        Parameters
        ----------
        model : sklearn.pipeline.Pipeline
            Model to be optimized.
        search_type : str, default="random"
            Type of search to perform. Can be "grid", "random", or "bayes".
        param_grid : dict, default=None
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such dictionaries,
            in which case the grids spanned by each dictionary in the list are
            explored. This enables searching over any sequence of parameter
            settings. Parameters names should be prefixed with "model__" to indicate that
            they are parameters of the model.
        niter : int, default=20
            Number of parameter settings that are sampled. Only used if
            search_type="random".

        Returns
        -------
        model : sklearn.pipeline.Pipeline
            Model pipeline with optimized parameters.
        """
        model_name = get_model_name(model)
        self.logger.info(f"Performing grid search for {model_name}...")

        # get default param grid if not provided
        if param_grid is None:
            param_grid = get_model_param_grid(model, search_type)
        # check that the provided param grid is valid
        else:
            param_grid = self.check_param_grid(param_grid, model)

        # adjust param_grid to include prefixes
        param_grid = adjust_param_grid(model, param_grid)

        # full grid search
        if search_type == "grid":
            # currently not available, give error message
            raise NotImplementedError("Grid search not available yet.")
            # searcher = GridSearchCV(
            #     model, param_grid, cv=self.cv, n_jobs=self.n_jobs, refit=True
            # )
        # random search
        elif search_type == "random":
            searcher = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=niter,
                cv=self.cv,
                n_jobs=self.n_jobs,
                refit=True,
            )
        # Bayes search || TODO, currently problems with skopt
        elif search_type == "bayes":
            searcher = BayesSearchCV(
                model,
                param_grid,
                n_iter=niter,
                cv=self.cv,
                n_jobs=self.n_jobs,
                refit=True,
            )

        searcher.fit(self.X, self.y)
        best_params = searcher.best_params_
        self.logger.info(f"Best parameters for {model_name}: {best_params}")

        return searcher

    @staticmethod
    def check_param_grid(param_grid, model):
        """Checks that the parameter grid is valid.

        Parameters
        ----------
        param_grid : dict, default=None
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such dictionaries,
            in which case the grids spanned by each dictionary in the list are
            explored. This enables searching over any sequence of parameter
            settings. Parameters names should be prefixed with "model__" to indicate that
            they are parameters of the model.
        model : sklearn.pipeline.Pipeline
            Model to be optimized.

        Returns
        -------
        param_grid : dict
        """
        if type(param_grid) != dict:
            raise TypeError("param_grid must be a dictionary")
        for key, value in param_grid.items():
            if type(key) != str:
                raise TypeError("param_grid keys must be strings")
            if type(value) != list:
                raise TypeError("param_grid values must be lists")

        inbuilt_grid = get_model_params(model)
        # check that all keys in param_grid are in the inbuilt grid
        for key in param_grid.keys():
            if key not in inbuilt_grid.keys():
                raise ValueError(f"Invalid parameter: {key}")

        return param_grid
