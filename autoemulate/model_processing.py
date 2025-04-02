"""Functions for getting and processing models."""
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from autoemulate.preprocess_target import get_dim_reducer
from autoemulate.preprocess_target import TargetPCA, NoChangeTransformer
from autoemulate.preprocess_target import VAEOutputPreprocessor


class ModelPrepPipeline:
    def __init__(
        self,
        model_registry,
        model_names,
        y,
        scale_input=False,
        scaler_input=None,
        reduce_dim_input=False,
        dim_reducer_input=None,
        scale_output=False,
        scaler_output=None,
        reduce_dim_output=False,
        dim_reducer_output=None,
    ):
        self.model_piped = None
        self.scaler_output = scaler_output
        self.transformer = PCA() #TODO: NoChangeTransformer() is not working 

        self.models = model_registry.get_models(model_names)

        self._turn_models_into_multioutput(y)

        self._wrap_model_reducer_in_pipeline(
            scale_input,
            scaler_input,
            reduce_dim_input,
            dim_reducer_input,
            scale_output,
            scaler_output,
            reduce_dim_output,
        )

    def _turn_models_into_multioutput(self, y):
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
        self.models_multi = [
            MultiOutputRegressor(model)
            if not model._more_tags().get("multioutput", False)
            and (y.ndim > 1 and y.shape[1] > 1)
            else model
            for model in self.models
        ]
        return self.models_multi

    def _wrap_model_reducer_in_pipeline(
        self,
        scale_input,
        scaler_input,
        reduce_dim_input,
        dim_reducer_input,
        scale_output,
        scaler_output,
        reduce_dim_output,
    ):
        """Wrap reducer in a pipeline if reduce_dim_output is True.

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
        self.models_piped = []

        for model in self.models_multi:
            input_steps = []
            if scale_input:
                input_steps.append(("scaler", scaler_input))
            if reduce_dim_input:
                input_steps.append(("dim_reducer", dim_reducer_input))
            input_steps.append(("model", model))
            input_pipeline = Pipeline(input_steps)

            # Create output transformation pipeline
            output_steps = []
            if scale_output:
                output_steps.append(("scaler_output", self.scaler_output))
            if reduce_dim_output:
                output_steps.append(("dim_reducer_output", self.transformer))

            if output_steps:
                output_pipeline = Pipeline(output_steps)
                final_model = TransformedTargetRegressor(
                    regressor=input_pipeline, transformer=output_pipeline
                )
                self.models_piped.append(final_model)
            else:
                # No output transformations, just use the input pipeline
                self.models_piped.append(input_pipeline)

        return self.models_piped
