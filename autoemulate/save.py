import json
import os
from pathlib import Path

import joblib
import numpy as np
import sklearn

from autoemulate.utils import get_model_name


class ModelSerialiser:
    def _save_model(self, model, path):
        """Saves a model + metadata to disk.

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

        # metadata
        meta = {
            "model": model_name,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        }
        with open(self._get_meta_path(path), "w") as f:
            json.dump(meta, f)

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
        meta_path = Path(self._get_meta_path(path))

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file {meta_path} not found.")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        sklearn_version = meta.get("scikit-learn", None)
        numpy_version = meta.get("numpy", None)

        for module, version, actual_version in [
            ("scikit-learn", sklearn_version, sklearn.__version__),
            ("numpy", numpy_version, np.__version__),
        ]:
            if version and version != actual_version:
                print(
                    f"Warning: {module} version mismatch. Expected {version}, found {actual_version}"
                )

        return model

    def _get_meta_path(self, path):
        """Returns the path to the metadata file.

        If the path has an extension, it is replaced with _meta.json.
        Otherwise, _meta.json is appended to the path.
        """
        path = Path(path)
        if path.suffix:
            meta_path = path.with_name(f"{path.stem}_meta.json")
        else:
            meta_path = path.with_name(path.name + "_meta.json")
        return meta_path
