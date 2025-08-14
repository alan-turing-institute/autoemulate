"""Functions for getting and processing models."""
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from autoemulate.preprocess_target import get_dim_reducer
from autoemulate.preprocess_target import InputOutputPipeline
from autoemulate.preprocess_target import NoChangeTransformer
from autoemulate.preprocess_target import TargetPCA
from autoemulate.preprocess_target import TargetVAE


class AutoEmulatePipeline:
    def __init__(
        self,
        model_registry,
        model_names,
        y,
        prep_config,
        scale_input=False,
        scaler_input=None,
        reduce_dim_input=False,
        dim_reducer_input=None,
        scale_output=False,
        scaler_output=None,
        reduce_dim_output=False,
    ):
        self.model_piped = None
        prep_name = prep_config["name"]
        prep_params = prep_config.get("params", {})
        self.dim_reducer_output = get_dim_reducer(prep_name, **prep_params)

        self.models = model_registry.get_models(model_names)

        self._turn_models_into_multioutput(y)

        # Store pipeline settings as instance attributes
        self.scale_input = scale_input
        self.scaler_input = scaler_input
        self.reduce_dim_input = reduce_dim_input
        self.dim_reducer_input = dim_reducer_input
        self.scale_output = scale_output
        self.scaler_output = scaler_output
        self.reduce_dim_output = reduce_dim_output

        # Wrap the model and reducer into a pipeline
        self._wrap_model_reducer_in_pipeline()

    def _wrap_model_reducer_in_pipeline(self):
        """Wrap reducer in a pipeline if reduce_dim_output is True."""
        self.models_piped = []

        for model in self.models_multi:
            input_steps = []
            if self.scale_input:
                input_steps.append(("scaler", self.scaler_input))
            if self.reduce_dim_input:
                input_steps.append(("dim_reducer", self.dim_reducer_input))
            input_steps.append(("model", model))
            input_pipeline = Pipeline(input_steps)

            # Create output transformation pipeline
            output_steps = []
            if self.scale_output:
                output_steps.append(("scaler_output", self.scaler_output))
            if self.reduce_dim_output:
                output_steps.append(("dim_reducer_output", self.dim_reducer_output))

            if output_steps:
                output_pipeline = Pipeline(output_steps)
                final_model = InputOutputPipeline(
                    regressor=input_pipeline, transformer=output_pipeline
                )
                self.models_piped.append(final_model)
            else:
                self.models_piped.append(input_pipeline)
        return self.models_piped

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
