"""Functions for getting and processing models."""
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from autoemulate.preprocess_target import get_dim_reducer
from autoemulate.preprocess_target import TargetPCA
from autoemulate.preprocess_target import VAEOutputPreprocessor


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

def _wrap_reducer_in_pipeline(models, scale_output, scaler_output, reduce_dim_output, dim_reducer_output):
    """Wrap reducer in a pipeline if reduce_dim_output is True. #TODO: should we pass "reduce_dim_output" bool at this point?

    Parameters
    ----------
    models : dict
        dict of model instances, with scaled models wrapped in a pipeline.
    dim_reducer_output : Reducer
        An instance of the Reducer class.

    Returns
    -------
    models_reduced : dict
        dict of model_names: model instances, with reduced models wrapped in a pipeline.
    """

    models_piped = []

    for models in models:
        # Retrieve the input pipeline
        input_pipeline = models

        # Create output transformation pipeline
        output_steps = []
        if scale_output:
            output_steps.append(("scaler_output", scaler_output))
        if reduce_dim_output:
            output_steps.append(("dim_reducer_output", dim_reducer_output))

        # If we have output transformations, wrap with TransformedTargetRegressor
        if output_steps:
            output_pipeline = Pipeline(output_steps)
            final_model = TransformedTargetRegressor(
                regressor=input_pipeline,
                transformer=output_pipeline
            )
            models_piped.append(final_model)
        else:
            # No output transformations, just use the input pipeline
            models_piped.append(input_pipeline)

    return models_piped

def _process_models(
    model_registry,
    model_names,
    y,
    scale,
    scaler,
    reduce_dim,
    dim_reducer,
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
    scale_output : bool
        Whether to scale the output data.
    scaler_output : sklearn.preprocessing object
        Scaler to use for outputs.
    reduce_dim_output : bool
        Whether to reduce the dimensionality of the output data.
    dim_reducer_output : sklearn.decomposition object
        Dimensionality reduction method to use for outputs.

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

def _process_reducers(
    models,
    scale_output,
    scaler_output,
    reduce_dim_output,
    dim_reducer_output
):
    """Process dimensionality reducers.

    Parameters
    ----------
    models : list
        List of model instances.
    reducer_output : Reducer
        An instance of the Reducer class.
    """
    #TODO: change "models_reduced" name
    models_reduced = _wrap_reducer_in_pipeline(
        models, scale_output, scaler_output, reduce_dim_output, dim_reducer_output
    )
    return models_reduced