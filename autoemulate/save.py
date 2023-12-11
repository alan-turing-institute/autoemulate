import os
import joblib
import json
import sklearn
import numpy as np
from autoemulate.utils import get_model_name


class ModelSerialiser:
    def save_model(self, model, path):
        """Saves a model + metadata to disk."""
        # Save model
        joblib.dump(model, path)

        # Create metadata
        meta = {
            "model": get_model_name(model),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        }

        # Save metadata
        with open(self.get_meta_path(path), "w") as f:
            json.dump(meta, f)

    def load_model(self, path):
        """Loads a model from disk and checks version."""
        # Load model
        model = joblib.load(path)

        # Create metadata file path
        meta_path = self.get_meta_path(path)

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file {meta_path} not found.")

        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)

        sklearn_version = meta.get("scikit-learn", None)
        numpy_version = meta.get("numpy", None)

        if sklearn_version and sklearn_version != sklearn.__version__:
            print(
                f"Warning: scikit-learn version mismatch. Expected {sklearn_version}, found {sklearn.__version__}"
            )

        if numpy_version and numpy_version != np.__version__:
            print(
                f"Warning: numpy version mismatch. Expected {numpy_version}, found {np.__version__}"
            )

        return model

    def get_meta_path(self, path):
        """Returns the path to the metadata file."""
        base, ext = os.path.splitext(path)
        # if path has extenstion, replace it with _meta.json
        # otherwise, append _meta.json
        if ext:
            meta_path = f"{base}_meta.json"
        else:
            meta_path = f"{base}_meta{ext}.json"

        return meta_path
