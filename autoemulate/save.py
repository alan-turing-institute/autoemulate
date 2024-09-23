import json
import os
from pathlib import Path

import joblib
import numpy as np
import sklearn

from autoemulate.utils import get_model_name


class ModelSerialiser:
    def _save_model(self, model, path):
        """Saves a model to disk.

        Parameters
        ----------
        model : scikit-learn model
            Model to save.
        models : dict
            Dictionary of model_name: model.
        path : str
            Path to save the model.
        """
        model_name = get_model_name(model)
        # check if path is directory
        if path is not None and Path(path).is_dir():
            path = Path(path) / model_name
        # save with model name if path is None
        if path is None:
            path = Path(model_name)
        else:
            path = Path(path)

        # create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # save model
        joblib.dump(model, path)

    def _save_models(self, models, path):
        """Saves all models

        Parameters
        ----------
        models : dict
            Dictionary of models.
        path : str
            Path to save the models.
        """
        if path is None:
            save_dir = Path.cwd()
        else:
            save_dir = Path(path)
            # create directory if it doesn't exist
            save_dir.parent.mkdir(parents=True, exist_ok=True)
        for model in models:
            model_path = save_dir / get_model_name(model)
            self._save_model(model, model_path)

    def _load_model(self, path):
        """Loads a model from disk and checks version."""
        path = Path(path)
        model = joblib.load(path)
        return model
