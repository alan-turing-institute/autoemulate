from sklearn.model_selection import GridSearchCV
import logging


class HyperparamSearch:
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
        logger : logging.Logger
            Logger object.
        """
        self.X = X
        self.y = y
        self.cv = cv
        self.n_jobs = n_jobs
        self.logger = logger or logging.getLogger(__name__)
        self.best_params = {}

    def search(self, model):
        """Performs hyperparameter search for a given model."""
        model_name = type(model.named_steps["model"]).__name__
        self.logger.info(f"Performing grid search for {model_name}...")

        try:
            param_grid = self.prepare_param_grid(model)
            grid_search = GridSearchCV(
                model, param_grid, cv=self.cv, n_jobs=self.n_jobs
            )
            grid_search.fit(self.X, self.y)

            best_params = self.extract_best_params(grid_search)
            self.logger.info(f"Best parameters for {model_name}: {best_params}")

            self.best_params = best_params
        except Exception as e:
            self.logger.error(f"Error during grid search for {model_name}: {e}")
            raise
        return model

    @staticmethod
    def prepare_param_grid(model):
        """Prepares the parameter grid with prefixed parameters."""
        param_grid = model.named_steps["model"].get_grid_params()
        return {f"model__{key}": value for key, value in param_grid.items()}

    @staticmethod
    def extract_best_params(grid_search):
        """Extracts and formats best parameters from the grid search."""
        best_params = grid_search.best_params_
        return {key.split("model__", 1)[1]: value for key, value in best_params.items()}
