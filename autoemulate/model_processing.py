"""Functions for getting and processing models."""
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from autoemulate.preprocess_target import get_dim_reducer, TargetPCA,VAEOutputPreprocessor


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
    dim_reducer : sklearn.decomposition object or str
        Dimensionality reduction method to use or its name.
    scale_output : bool, default=False
        Whether to scale the output data.
    reduce_dim_output : bool, default=False
        Whether to reduce the dimensionality of the output data.
    scaler_output : sklearn.preprocessing object
        Scaler to use for outputs. If None and scale_output is True, a StandardScaler will be used.
    dim_reducer_output : sklearn.decomposition object or str
        Dimensionality reduction method to use for outputs or its name.
        If None and reduce_dim_output is True, a PCA will be used.  #TODO

    Returns
    -------
    models_scaled : dict
        dict of model_names: model instances, with scaled models wrapped in a pipeline.
    """
    models_piped = []
    for model in models:
        steps = []
        if reduce_dim_output:
           # Only call get_dim_reducer if dim_reducer is a string
            reducer = get_dim_reducer(name=dim_reducer_output)
            steps.append(
                ("Dimentionality reducer for output ",reducer)
            )

        print(reduce_dim,reduce_dim_output,reducer)

        # Add X preprocessing steps
        if scale:
            steps.append(("scaler", scaler))
        if reduce_dim:
            steps.append(("dim_reducer", dim_reducer))

        # Add the model as the final step
        steps.append(("model", model))

        # Use YAwarePipeline if we need to transform y, otherwise use standard Pipeline
        if reduce_dim_output:
            pipeline = AutoEmulatePipeline(steps)
        else:
            pipeline = Pipeline(steps)

        # Explicitly ensure model_name is set
        if hasattr(model, "model_name"):
            pipeline.model_name = model.model_name

        models_piped.append(pipeline)
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





class AutoEmulatePipeline(Pipeline):
    """Pipeline that can modify both X and y, while preserving all Pipeline attributes."""

    def __init__(self, steps, memory=None, verbose=True):
        # Initialize with parent class - this will set up all the Pipeline internals
        super().__init__(steps=steps, memory=memory, verbose=verbose)

        # Track the steps separately to avoid property issues
        self._y_transformers = []
        self._x_steps = []


        # Process the steps after super().__init__() has been called
        for name, transform in steps:
            # Set verbosity on each transformer if it has the attribute
            if hasattr(transform, 'verbose'):
                transform.verbose = verbose

            if isinstance(transform, TargetPCA):
                self._y_transformers.append((name, transform))
            else:
                self._x_steps.append((name, transform))

        # Propagate model_name from final estimator
        if hasattr(self._final_estimator, "model_name"):
            self.model_name = self._final_estimator.model_name

        # Create a standard pipeline for X transformations
        if self._x_steps:
            self.x_pipeline = Pipeline(self._x_steps)
            # Also propagate model_name to this sub-pipeline
            if hasattr(self, "model_name"):
                self.x_pipeline.model_name = self.model_name

    def fit(self, X, y=None, **fit_params):
        """Apply y transformations first, then fit X pipeline."""
        X_temp, y_temp = X, y

        # Apply y transformations if any
        for _, y_transformer in self._y_transformers:
            y_transformer.fit(X_temp, y_temp)
            X_temp, y_temp = y_transformer.transform(X_temp, y_temp)

        # Then fit the X pipeline or the parent Pipeline
        if hasattr(self, "x_pipeline"):
            self.x_pipeline.fit(X_temp, y_temp, **fit_params)
        else:
            super().fit(X_temp, y_temp, **fit_params)

        return self

    def predict(self, X, **predict_params):
        """Predict using the X pipeline or parent Pipeline."""
        # Handle `return_std` and similar parameters
        return_std = predict_params.pop("return_std", False)

        if hasattr(self, "x_pipeline"):
            # Pass parameters to the underlying model's predict method
            if return_std:
                y_pred, y_std = self.x_pipeline.predict(
                    X, return_std=return_std, **predict_params
                )
            else:
                y_pred = self.x_pipeline.predict(X, **predict_params)
        else:
            if return_std:
                y_pred, y_std = super().predict(
                    X, return_std=return_std, **predict_params
                )
            else:
                y_pred = super().predict(X, **predict_params)

        # Inverse transform y predictions through y transformers in reverse order
        X_temp, y_temp = X, y_pred
        for _, y_transformer in reversed(self._y_transformers):
            X_temp, y_temp = y_transformer.inverse_transform(X_temp, y_temp)

        # Return results with or without std based on `return_std`
        if return_std:
            return y_temp, y_std
        return y_temp

    def transform(self, X):
        """Transform X using the X pipeline or parent Pipeline."""
        if hasattr(self, "x_pipeline"):
            return self.x_pipeline.transform(X)
        return super().transform(X)

    def score(self, X, y=None, sample_weight=None):
        """Transform y before scoring."""
        X_temp, y_temp = X, y

        # Transform y for scoring
        for _, y_transformer in self._y_transformers:
            X_temp, y_temp = y_transformer.transform(X_temp, y_temp)

        # Then score using the X pipeline or parent Pipeline
        if hasattr(self, "x_pipeline"):
            return self.x_pipeline.score(X_temp, y_temp, sample_weight)
        return super().score(X_temp, y_temp, sample_weight)