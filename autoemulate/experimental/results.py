from typing import Any

import pandas as pd

from autoemulate.experimental.emulators.transformed.base import TransformedEmulator


class Result:
    def __init__(  # noqa: PLR0913
        self,
        id: str,
        model_name: str,
        model: TransformedEmulator,
        config: dict[str, Any],
        r2_score: float,
        rmse_score: float,
    ):
        self.id = id
        self.model_name = model_name
        self.model = model
        self.x_transforms = model.x_transforms
        self.y_transforms = model.y_transforms
        self.config = config
        self.r2_score = r2_score
        self.rmse_score = rmse_score


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
            ['id', 'model', 'x_transforms', 'y_transforms', 'config', 'r2_score',
             'rmse_score'].
        TODO: include test data
        """
        data = {
            "id": [result.id for result in self.results],
            "model_name": [result.model_name for result in self.results],
            "x_transforms": [result.x_transforms for result in self.results],
            "y_transforms": [result.y_transforms for result in self.results],
            "rmse_score": [result.rmse_score for result in self.results],
            "r2_score": [result.r2_score for result in self.results],
        }
        df = pd.DataFrame(data)
        return df.sort_values(by="r2_score", ascending=False)

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
        return max(self.results, key=lambda r: r.r2_score)

    def get_result(self, result_id: str) -> Result:
        """
        Get a result by its ID.
        Parameters
        ----------
        result_id: str
            The ID of the model to retrieve.
        Returns
        -------
        Result: The result with the specified ID.
        """
        try:
            return self._id_to_result[result_id]
        except KeyError as err:
            raise ValueError(f"No result found with ID: {result_id}") from err
