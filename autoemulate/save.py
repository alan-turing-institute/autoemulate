import json
import os

import joblib
import numpy as np
import sklearn

from autoemulate.utils import get_model_name


class ModelSerialiser:
    # TODO: Add model type, is Pipeline correct?
    def save_model(self, model, path: str) -> None:
        """Saves a model + metadata to disk.

        Parameters
        ----------
        model : object
            Model to save.
        path : str
            Path to save the model to.

        Returns
        -------
        None
        """
        # model
        joblib.dump(model, path)

        # metadata
        meta = {
            "model": get_model_name(model),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        }

        with open(self.get_meta_path(path), "w") as f:
            json.dump(meta, f)

    # TODO: Add return type, is Pipeline correct?
    def load_model(self, path: str):
        """Loads a model from disk and checks version.

        Parameters
        ----------
        path : str
            Path to the model file.

        Returns
        -------
        <TYPE>
            Model instance.
        """
        model = joblib.load(path)
        meta_path = self.get_meta_path(path)

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

    def get_meta_path(self, path: str) -> str:
        """Returns the path to the metadata file.

        If the path has an extension, it is replaced with _meta.json.
        Otherwise, _meta.json is appended to the path.

        Parameters
        ----------
        path : str
            Path to the model file.

        Returns
        -------
        str
            Path to the metadata file.
        """
        base, ext = os.path.splitext(path)
        meta_path = f"{base}_meta.json" if ext else f"{base}_meta{ext}.json"
        return meta_path
