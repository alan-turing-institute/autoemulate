"""Functions for getting and processing models."""
from copy import deepcopy

from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from autoemulate.utils import get_model_name
from autoemulate.utils import get_short_model_name


def _get_models(model_registry, model_subset=None):
    """Get models from REGISTRY.
    Takes a subset of models if model_subset argument was used in setup().

    Parameters
    ----------
    model_registry : dict
        Dictionary of models.
    model_subset : list
        List of model names.

    Returns
    -------
    dict
        Dictionary of models instances.
    """
    # TODO replace deepcopy with a proper model register
    if model_subset is not None:
        _check_model_names(model_subset, model_registry)
        models = [deepcopy(model_registry)[model] for model in model_subset]
    else:
        models = list(deepcopy(model_registry).values())
    return models


def _check_model_names(model_names, model_registry):
    """Check whether model_names are in MODEL_REGISTRY

    Parameters
    ----------
    model_names : list
        List of model names.
    model_registry : dict
        Dictionary of models.

    Returns
    -------
    None
        Raises ValueError if a model in chosen_models is not in MODEL_REGISTRY.
    """
    for model in model_names:
        if model not in model_registry:
            raise ValueError(
                f"Model {model} not found. Available models are: {model_registry.keys()}"
            )


def _turn_models_into_multioutput(models, y):
    """Turn single output models into multioutput models if y is 2D.

    Parameters
    ----------
    models : dict
        Dict of model instances.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.

    Returns
    -------
    models_multi : dict
        Dict with model instances, where single output models are now wrapped in MultiOutputRegressor.
    """

    models_multi = [
        MultiOutputRegressor(model)
        if not model._more_tags()["multioutput"] and (y.ndim > 1 and y.shape[1] > 1)
        else model
        for model in models
    ]
    return models_multi


def _wrap_models_in_pipeline(models, scale, scaler, reduce_dim, dim_reducer):
    """Wrap models in a pipeline if scale is True.

    Parameters
    ----------
    models : dict
        dict of model instances.
    scale : bool
        Whether to scale the data.
    scaler : sklearn.preprocessing object
        Scaler to use.
    reduce_dim : bool
        Whether to reduce the dimensionality of the data.
    dim_reducer : sklearn.decomposition object
        Dimensionality reduction method to use.

    Returns
    -------
    models_scaled : dict
        dict of model_names: model instances, with scaled models wrapped in a pipeline.
    """

    models_piped = []

    for model in models:
        steps = []
        if scale:
            steps.append(("scaler", scaler))
        if reduce_dim:
            steps.append(("dim_reducer", dim_reducer))
        steps.append(("model", model))
        # without scaling or dim reduction, the model is the only step
        models_piped.append(Pipeline(steps))

    return models_piped


def _get_and_process_models(
    model_registry, model_subset, y, scale, scaler, reduce_dim, dim_reducer
):
    """Get and process models.

    Parameters
    ----------
    model_registry : dict
        Dictionary of models.
    model_subset : list
        List of model names.
    y : array-like, shape (n_samples, n_outputs)
        Simulation output.
    scale : bool
        Whether to scale the data.
    scaler : sklearn.preprocessing object
        Scaler to use.

    Returns
    -------
    models : list
        List of model instances.
    """
    models = _get_models(model_registry, model_subset)
    models_multi = _turn_models_into_multioutput(models, y)
    models_scaled = _wrap_models_in_pipeline(
        models_multi, scale, scaler, reduce_dim, dim_reducer
    )
    return models_scaled


def _get_model_names(models):
    """Get model names (short and long) from models.

    Parameters
    ----------
    models : list
        List of model instances.

    Returns
    -------
    model_names_dict : dict
        Dictionary of model names (long and short).
    """
    model_names = [get_model_name(model) for model in models]
    model_names_short = [get_short_model_name(model) for model in models]
    model_names_dict = dict(zip(model_names, model_names_short))
    return model_names_dict
