import pandas as pd

from autoemulate.utils import get_short_model_name

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

if _in_ipython_session:
    from IPython.display import display, HTML


def _display_results(title, content):
    """Helper function to display results based on environment."""
    content = content.round(4)

    if _in_ipython_session:
        from IPython.display import display, HTML

        display(HTML(f"<p>{title}</p>"))
        display(HTML(content.to_html()))
    else:
        print(title)
        print(content)


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

    settings = pd.DataFrame(
        [
            str(cls.X.shape),
            str(cls.y.shape),
            str(cls.test_set_size),
            str(cls.scale),
            str(cls.scaler.__class__.__name__ if cls.scaler is not None else "None"),
            str(cls.param_search),
            str(cls.search_type),
            str(cls.param_search_iters),
            str(cls.reduce_dim),
            str(
                cls.dim_reducer.__class__.__name__
                if cls.dim_reducer is not None
                else "None"
            ),
            str(
                cls.cross_validator.__class__.__name__
                if cls.cross_validator is not None
                else "None"
            ),
            str(cls.n_jobs if cls.n_jobs is not None else "1"),
        ],
        index=[
            "Simulation input shape (X)",
            "Simulation output shape (y)",
            "Proportion of data for testing (test_set_size)",
            "Scale input data (scale)",
            "Scaler (scaler)",
            "Do hyperparameter search (param_search)",
            "Type of hyperparameter search (search_type)",
            "Number of sampled parameter settings (param_search_iters)",
            "Reduce dimensionality (reduce_dim)",
            "Dimensionality reduction method (dim_reducer)",
            "Cross validator (cross_validator)",
            "Parallel jobs (n_jobs)",
        ],
        columns=["Values"],
    )

    # if cls.param_search == False, remove the search_type and param_search_iters rows
    if not cls.param_search:
        settings = settings.drop(
            [
                "Type of hyperparameter search (search_type)",
                "Number of sampled parameter settings (param_search_iters)",
            ]
        )

    # if cls.reduce_dim == False, remove the dim_reducer row
    if not cls.reduce_dim:
        settings = settings.drop(["Dimensionality reduction method (dim_reducer)"])

    # if cls.scale == False, remove the scaler row
    if not cls.scale:
        settings = settings.drop(["Scaler (scaler)"])

    settings_str = settings.to_string(index=True, header=False)
    width = len(settings_str.split("\n")[0])

    if _in_ipython_session:
        display(HTML("<p>AutoEmulate is set up with the following settings:</p>"))
        display(HTML(settings.to_html()))
        return

    # when not in a notebook, print the settings in a table
    print("AutoEmulate is set up with the following settings:")
    print("-" * width)
    print(settings_str)
    print("-" * width)
