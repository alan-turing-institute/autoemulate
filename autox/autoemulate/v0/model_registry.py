from autoemulate.utils import get_short_model_name


class ModelRegistry:
    """
    A registry for emulator models.
    """

    def __init__(self):
        self.models = {}
        self.core_model_names = []

    def register_model(self, model_name, model_class, is_core=False):
        self.models[model_name] = model_class
        if is_core:
            self.core_model_names.append(model_name)

    def get_model_names(self, models=None, is_core=False):
        """Get a dictionary of (all) model names and their short names

        Parameters
        ----------
        models : str or list of str
            The name(s) of the model(s) to get long and short names for.
        is_core : bool
            Whether to return only core model names in case `models` is None.

        Returns
        -------
        dict
            A dictionary of model names and their short names.
        """

        model_names = {
            model_name: get_short_model_name(model())
            for model_name, model in self.models.items()
        }

        if models is not None:
            # if models is a string, convert it to a list
            if isinstance(models, str):
                models = [models]

            # if not, check that it is a list
            if not isinstance(models, list):
                raise ValueError(
                    f"models must be a list of model names. Got {models} of type {type(models)}"
                )
            # check that all model names are valid, either as key (long name) or value (short name) of model_names
            if not all(
                model_name in model_names or model_name in model_names.values()
                for model_name in models
            ):
                raise ValueError(
                    f"One or more model names in {models} not found. Available models: {', '.join(model_names.keys())} or short names: {', '.join(model_names.values())}"
                )
            # models is a list with model names.
            # They can be short or long names, so either key or value of model_names.
            # Subset the model_names dict.
            model_names = {
                k: v for k, v in model_names.items() if k in models or v in models
            }

        if models is None and is_core:
            model_names = {
                k: v for k, v in model_names.items() if k in self.core_model_names
            }

        return model_names

    def get_core_models(self):
        """Get a list of initialized core models"""
        return [self.models[core_model]() for core_model in self.core_model_names]

    def get_all_models(self):
        """Get a list of all models, and initialize them"""
        return [self.models[model_name]() for model_name in self.models.keys()]

    def get_models(self, models=None):
        """Get a list of subset of models, default is core models

        Parameters
        ----------
        models : str or list of str
            The name(s) of the model(s) to get. Can be long or short names.
        """
        if models is None:
            return self.get_core_models()
        else:
            model_names = self.get_model_names(models)
            return [
                self.models[model_name]()
                for model_name in model_names or model_names.values()
            ]
