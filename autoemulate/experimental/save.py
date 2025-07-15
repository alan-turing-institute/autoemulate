from pathlib import Path

import joblib

from autoemulate.experimental.emulators.base import Emulator


class ModelSerialiser:
    def __init__(self, logger):
        self.logger = logger

    def _save_model(
        self, model: Emulator, model_name: str | None, path: str | Path | None = None
    ):
        """Saves a model to disk.

        Parameters
        ----------
        model : Emulator model
            Model to save.
        model_name : str or None, optional
            Name of the model to save.
            If None, the model's name will be used.
        path : str, Path or None, optional
            Path to save the model.
            If None, the model will be saved with the model name.
        """
        if model_name is None:
            model_name = self._get_model_name(model)
        full_path = self._prepare_path(path, model_name)

        try:
            joblib.dump(model, full_path)
            self.logger.info("%s saved to %s", model_name, full_path)
        except Exception as e:
            self.logger.error("Failed to save %s to %s: %s", model_name, full_path, e)
            raise

    def _load_model(self, path: str | Path):
        """Loads a model from disk.

        Parameters
        ----------
        path : str or Path
            Path to load the model.
        """
        path = Path(path)
        try:
            model = joblib.load(path)
            self.logger.info("Model loaded from %s", path)
            return model
        except Exception as e:
            self.logger.error("Failed to load model from %s: %s", path, e)
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
        model_name = model.model_name()
        return f"{model_name}.joblib"
