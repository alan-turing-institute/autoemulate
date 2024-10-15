"""Functions for getting and processing models."""
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline


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


def _process_models(
    model_registry, model_names, y, scale, scaler, reduce_dim, dim_reducer
):
    """Get and process models.

    Parameters
    ----------
    model_registry : ModelRegistry
        An instance of the ModelRegistry class.
    model_names : list
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
    models = model_registry.get_models(model_names)
    models_multi = _turn_models_into_multioutput(models, y)
    models_scaled = _wrap_models_in_pipeline(
        models_multi, scale, scaler, reduce_dim, dim_reducer
    )
    return models_scaled
