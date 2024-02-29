import pandas as pd

from autoemulate.utils import get_mean_scores
from autoemulate.utils import get_model_name
from autoemulate.utils import get_model_scores

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

if _in_ipython_session:
    from IPython.display import display, HTML


def _print_cv_results(models, scores_df, model=None, sort_by="r2"):
    """Print cv results.

    Parameters
    ----------
    models : list
        A list of models.
    scores_df : pandas.DataFrame
        A dataframe with scores for each model, metric, and fold.
    model : str, optional
        The name of the model to print. If None, the best fold from each model will be printed.
        If a model name is provided, the scores for that model across all folds will be printed.
    sort_by : str, optional
        The metric to sort by. Default is "r2".

    """
    # check if model is in self.models
    if model is not None:
        model_names = [get_model_name(model) for model in models]
        if model not in model_names:
            raise ValueError(
                f"Model {model} not found. Available models are: {model_names}"
            )
    if model is None:
        means = get_mean_scores(scores_df, metric=sort_by)
        print("Average scores across all models:")
        print(means)
    else:
        scores = get_model_scores(scores_df, model)
        print(f"Scores for {model} across all folds:")
        print(scores)


def _print_setup(cls):
    """Print the setup of the AutoEmulate object.

    If in an IPython session, the setup will be displayed as an HTML table.

    Parameters
    ----------
    cls : AutoEmulate
        The AutoEmulate object.
    """
    if not cls.is_set_up:
        raise RuntimeError("Must run setup() before print_setup()")

    models = "\n- " + "\n- ".join(
        [
            x[1].__class__.__name__
            for pipeline in cls.models
            for x in pipeline.steps
            if x[0] == "model"
        ]
    )
    metrics = "\n- " + "\n- ".join([metric.__name__ for metric in cls.metrics])

    settings = pd.DataFrame(
        [
            str(cls.X.shape),
            str(cls.y.shape),
            str(cls.train_idxs.shape[0]),
            str(cls.test_idxs.shape[0]),
            str(cls.param_search),
            str(cls.search_type),
            str(cls.param_search_iters),
            str(cls.scale),
            str(cls.scaler.__class__.__name__ if cls.scaler is not None else "None"),
            str(cls.reduce_dim),
            str(
                cls.dim_reducer.__class__.__name__
                if cls.dim_reducer is not None
                else "None"
            ),
            str(cls.cv.__class__.__name__ if cls.cv is not None else "None"),
            str(cls.folds),
            str(cls.n_jobs if cls.n_jobs is not None else "1"),
        ],
        index=[
            "Simulation input shape (X)",
            "Simulation output shape (y)",
            "# training set samples (train_idxs)",
            "# test set samples (test_idxs)",
            "Do hyperparameter search (param_search)",
            "Type of hyperparameter search (search_type)",
            "# sampled parameter settings (param_search_iters)",
            "Scale data before fitting (scale)",
            "Scaler (scaler)",
            "Dimensionality reduction before fitting (reduce_dim)",
            "Dimensionality reduction method (dim_reducer)",
            "Cross-validation strategy (cv)",
            "# folds (folds)",
            "# parallel jobs (n_jobs)",
        ],
        columns=["Values"],
    )

    settings_str = settings.to_string(index=True, header=False)
    width = len(settings_str.split("\n")[0])

    if _in_ipython_session:
        display(HTML("<p>AutoEmulate is set up with the following settings:</p>"))
        display(HTML(settings.to_html()))
        return

    print("AutoEmulate is set up with the following settings:")
    print("-" * width)
    print(settings_str)
    print("-" * width)
    print("Models:" + models)
    print("-" * width)
    print("Metrics:" + metrics)
    print("-" * width)
