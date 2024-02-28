import json
import os

import joblib
import numpy as np
import sklearn

from autoemulate.utils import get_model_name


class ModelSerialiser:
    def _save_model(self, model, path):
        """Saves a model + metadata to disk."""

        # check if path is directory
        if path is not None and os.path.isdir(path):
            model_name = get_model_name(model)
            path = os.path.join(path, model_name)
        # save with model name if path is None
        if path is None:
            path = get_model_name(model)

        # model
        joblib.dump(model, path)

        # metadata
        meta = {
            "model": get_model_name(model),
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
        if path is not None and not os.path.isdir(path):
            raise ValueError("Path must be a directory")
        for model in models.values():
            self._save_model(model, path)

    def _load_model(self, path):
        """Loads a model from disk and checks version."""
        model = joblib.load(path)
        meta_path = self._get_meta_path(path)

        if not os.path.exists(meta_path):
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
        base, ext = os.path.splitext(path)
        meta_path = f"{base}_meta.json" if ext else f"{base}_meta{ext}.json"
        return meta_path
