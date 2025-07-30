import ast
from pathlib import Path

import joblib
import pandas as pd

from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.results import Result  # , Results


class ModelSerialiser:
    """ModelSerialiser handles saving and loading of models and results."""

    def __init__(self, logger):
        self.logger = logger

    def _save_model(
        self,
        model: Emulator,
        model_name: str | None = None,
        path: str | Path | None = None,
    ):
        """Save a model to disk.

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
        model_filename = self._get_model_filename(model, model_name)
        full_path = self._prepare_path(path, model_filename)
        try:
            if full_path.suffix != ".joblib":
                joblib.dump(model, full_path.with_suffix(".joblib"))
            else:
                joblib.dump(model, full_path)
            self.logger.info("%s saved to %s", model_filename, full_path)
            return full_path
        except Exception as e:
            self.logger.error(
                "Failed to save %s to %s: %s", model_filename, full_path, e
            )
            raise

    def _save_result(
        self,
        result: Result,
        result_name: str | None = None,
        path: str | Path | None = None,
    ) -> Path:
        """Save a result and model to disk.

        Parameters
        ----------
        result : Result
            Result object containing the model and its metadata.
        result_name : str or None, optional
            Name of the result to save. If None, the result name and id will be used.
        path : str, Path or None, optional
            Path to save the model and metadata.
            If None, the result will be saved with the result_name.

        Returns
        -------
        Path
            Full path of saved files without either the .joblib or _metadata.csv suffix.
        """
        if result_name is None:
            result_name = f"{result.model_name}_{result.id}"
        full_path = self._prepare_path(path, result_name)

        self._save_model(result.model, result_name, full_path)

        # Save metadata to CSV
        metadata_path = Path(f"{full_path}_metadata.csv")
        metadata_df = result.metadata_df()
        try:
            metadata_df.to_csv(metadata_path, index=False)
            self.logger.info("Metadata saved to %s", metadata_path)
        except Exception as e:
            self.logger.error("Failed to save metadata to %s: %s", metadata_path, e)
            raise

        return full_path

    def _load_model(self, path: str | Path):
        """Load a model from disk.

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

    def _load_result(self, path: str | Path) -> Result | Emulator:
        """
        Load a result or model from disk.

        Load a model and (if it exists) its metadata from disk,
        returning either a Result or Emulator object.

        Parameters
        ----------
        path : str or Path
            Path to the model file.

        Returns
        -------
        Result or Emulator
            The loaded model or result object.
        """
        if ".joblib" not in str(path):
            model_path = Path(f"{path}.joblib")
        else:
            model_path = Path(path)
        model = self._load_model(model_path)
        metadata_path = Path(f"{path}_metadata.csv")
        try:
            metadata_df = pd.read_csv(metadata_path, nrows=1)
            self.logger.info("Metadata loaded from %s", metadata_path)
        except Exception as e:
            msg = "Failed to load metadata from %s: %s"
            self.logger.error(msg, metadata_path, e)
            return model
        row = metadata_df.iloc[0]
        params = row["params"]
        params = ast.literal_eval(params)

        return Result(
            id=row["id"],
            model_name=row["model_name"],
            model=model,
            params=params,
            r2_test=row["r2_test"],
            rmse_test=row["rmse_test"],
            r2_test_std=row["r2_test_std"],
            rmse_test_std=row["rmse_test_std"],
            r2_train=row["r2_train"],
            rmse_train=row["rmse_train"],
            r2_train_std=row["r2_train_std"],
            rmse_train_std=row["rmse_train_std"],
        )

    def _prepare_path(self, path: str | Path | None, model_name: str):
        """Prepare path for saving model."""
        if path is None:
            full_path = Path(model_name)
        else:
            full_path = Path(path)
            if full_path.is_dir():
                full_path = full_path / model_name

        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    @staticmethod
    def _get_model_filename(model, provided_name: str | None = None) -> str:
        """Get the name of the model to save."""
        model_name = provided_name if provided_name is not None else model.model_name()
        return f"{model_name}.joblib"
