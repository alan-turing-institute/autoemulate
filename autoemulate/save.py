import json
import os
from pathlib import Path

import joblib
import numpy as np
import sklearn

from autoemulate.utils import get_model_name


class ModelSerialiser:
    def __init__(self, logger):
        self.logger = logger

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
        logger : logging.Logger
            Logger to use.
        """
        model_name = get_model_name(model)
        path = self._prepare_path(path, model_name)
        try:
            joblib.dump(model, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save model to {path}: {e}")
            raise

        # save model
        joblib.dump(model, path)

    def _load_model(self, path):
        """Loads a model from disk and checks version."""
        path = Path(path)
        try:
            model = joblib.load(path)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model from {path}: {e}")
            raise
        return model

    def _prepare_path(self, path, model_name):
        """Prepares path for saving model."""
        if path is not None and Path(path).is_dir():
            path = Path(path) / model_name
        # save with model name if path is None
        if path is None:
            path = Path(model_name)
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        return path
