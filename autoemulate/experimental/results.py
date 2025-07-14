from typing import Any

import pandas as pd

from autoemulate.experimental.emulators.transformed.base import TransformedEmulator


class Result:
    def __init__(  # noqa: PLR0913
        self,
        id: int,
        model_name: str,
        model: TransformedEmulator,
        config: dict[str, Any],
        r2_test: float,
        rmse_test: float,
        r2_train: float,
        rmse_train: float,
        r2_test_std: float,
        rmse_test_std: float,
        r2_train_std: float,
        rmse_train_std: float,
    ):
        self.id = id
        self.model_name = model_name
        self.model = model
        self.x_transforms = model.x_transforms
        self.y_transforms = model.y_transforms
        self.config = config
        self.r2_test = r2_test
        self.rmse_test = rmse_test
        self.r2_test_std = r2_test_std
        self.rmse_test_std = rmse_test_std
        self.r2_train = r2_train
        self.rmse_train = rmse_train
        self.r2_train_std = r2_train_std
        self.rmse_train_std = rmse_train_std


class Results:
    def __init__(
        self,
        results: list[Result] | None = None,
    ):
        if results is None:
            results = []
        self.results = results
        self._id_to_result = {result.id: result for result in self.results}

    def _update_index(self):
        self._id_to_result = {result.id: result for result in self.results}

    def add_result(self, result: Result):
        self.results.append(result)
        self._update_index()

    def summarize(self) -> pd.DataFrame:
        """
        Summarize the results in a DataFrame.
        Returns:
            pd.DataFrame: DataFrame with columns:
            ['model', 'x_transforms', 'y_transforms', 'config', 'r2_score',
             'rmse_score'].
        TODO: include test data
        """
        data = {
            "model_name": [result.model_name for result in self.results],
            "x_transforms": [result.x_transforms for result in self.results],
            "y_transforms": [result.y_transforms for result in self.results],
            "config": [result.config for result in self.results],
            "rmse_test": [result.rmse_test for result in self.results],
            "r2_test": [result.r2_test for result in self.results],
            "r2_test_std": [result.r2_test_std for result in self.results],
            "r2_train": [result.r2_train for result in self.results],
            "r2_train_std": [result.r2_train_std for result in self.results],
        }
        df = pd.DataFrame(data)
        return df.sort_values(by="r2_test", ascending=False)

    summarise = summarize

    def best_result(self) -> Result:
        """
        Get the model with the best result based on the highest R2 score.
        Returns:
            Result: The result with the highest R2 score.
        """
        if not self.results:
            msg = "No results available. Please run AutoEmulate.compare() first."
            raise ValueError(msg)
        return max(self.results, key=lambda r: r.r2_test)

    def get_result(self, result_id: int) -> Result:
        """
        Get a result by its ID.
        Parameters
        ----------
        result_id: int
            The ID of the model to retrieve.
        Returns
        -------
        Result: The result with the specified ID.
        """
        try:
            return self._id_to_result[result_id]
        except KeyError as err:
            raise ValueError(f"No result found with ID: {result_id!s}") from err
