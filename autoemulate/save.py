from pathlib import Path

import joblib

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
        path : str, Path or None, optional
            Path to save the model. If None, the model will be saved with the model name.
        """
        model_name = self._get_model_name(model)
        full_path = self._prepare_path(path, model_name)

        try:
            joblib.dump(model, full_path)
            self.logger.info(f"{model_name} saved to {full_path}")
        except Exception as e:
            self.logger.error(f"Failed to save {model_name} to {full_path}: {e}")
            raise

    def _load_model(self, path):
        """Loads a model from disk.

        Parameters
        ----------
        path : str or Path
            Path to load the model.
        """
        path = Path(path)
        try:
            model = joblib.load(path)
            self.logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model from {path}: {e}")
            raise

    def _prepare_path(self, path, model_name):
        """Prepares path for saving model."""
        if path is None:
            full_path = Path(model_name)
        else:
            full_path = Path(path)
            if full_path.is_dir():
                full_path = full_path / model_name

        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    @staticmethod
    def _get_model_name(model):
        """Gets the name of the model to save."""
        model_name = get_model_name(model)
        return f"{model_name}.joblib"
