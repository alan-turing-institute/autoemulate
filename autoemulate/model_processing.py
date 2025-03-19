"""Functions for getting and processing models."""
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from autoemulate.preprocess_target import AutoencoderDimReducer, VariationalAutoencoderDimReducer

def _turn_models_into_multioutput(models, y):
    """Turn single output models into multioutput models.
    
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


def _wrap_models_in_pipeline(
    models, scale, scaler, reduce_dim, dim_reducer, scale_output=False, reduce_dim_output=False, 
    scaler_output=None, dim_reducer_output=None
):
    """Wrap models in a pipeline.

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
    scale_output : bool, default=False
        Whether to scale the output data.
    reduce_dim_output : bool, default=False
        Whether to reduce the dimensionality of the output data.
    scaler_output : sklearn.preprocessing object
        Scaler to use for outputs. If None and scale_output is True, a StandardScaler will be used.
    dim_reducer_output : sklearn.decomposition object
        Dimensionality reduction method to use for outputs. If None and reduce_dim_output is True, a PCA will be used.

    Returns
    -------
    models_scaled : dict
        dict of model_names: model instances, with scaled models wrapped in a pipeline.
    """

    models_piped = []

    for model in models:
        # Create input transformation pipeline
        input_steps = []
        if scale:
            input_steps.append(("scaler", scaler))
        if reduce_dim:
            input_steps.append(("dim_reducer", dim_reducer))
        
        # Create input transformation pipeline
        input_steps.append(("model", model))
        input_pipeline = Pipeline(input_steps)

        # Create output transformation pipeline
        output_steps = []
        if scale_output:
            output_steps.append(("scaler", scaler_output))
        if reduce_dim_output:
            if dim_reducer_output == 'AE':
                dim_reducer_output = AutoencoderDimReducer(
                                        encoding_dim=8,           # Dimensionality of the encoded space
                                        hidden_layers=[64, 32],    # Architecture of the encoder (decoder will be symmetric)
                                        epochs=1000,
                                        batch_size=32
                                    )
            elif dim_reducer_output == 'VAE':
                dim_reducer_output = VariationalAutoencoderDimReducer(
                                encoding_dim=8,         # Dimension of latent space
                                hidden_layers=[64, 32],  # Hidden layer dimensions
                                epochs=1000,
                                batch_size=32,
                                beta=1.0,                # Weight for KL divergence term
                                verbose=True             # Show training progress
                            )

            output_steps.append(("dim_reducer", dim_reducer_output))

       # If we have output transformations, wrap with TransformedTargetRegressor
        if output_steps:
            output_transformer = Pipeline(output_steps)
            final_model = TransformedTargetRegressor(
                regressor=input_pipeline,
                transformer=output_transformer
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
    scale_output,
    scaler_output,
    reduce_dim_output,
    dim_reducer_output
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
        models_multi, 
        scale, 
        scaler, 
        reduce_dim, 
        dim_reducer,
        scale_output, 
        reduce_dim_output, 
        scaler_output, 
        dim_reducer_output
    )
    return models_scaled
