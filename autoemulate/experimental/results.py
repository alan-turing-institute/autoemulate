from typing import Any

import pandas as pd

from autoemulate.experimental.emulators.base import Emulator


class Result:
    def __init__(
        self,
        id: str,
        model: Emulator,
        config: dict[str, Any],
        r2_score: float,
        rmse_score: float,
    ):
        self.id = id
        self.model = model
        self.config = config
        self.r2_score = r2_score
        self.rmse_score = rmse_score

    # Equiv of plot_eval
    # def plot(self, ...):


class Results:
    def __init__(
        self,
        results: list[Result] | None = None,
    ):
        if results is None:
            results = []
        self.results = results

    def summarize(self) -> pd.DataFrame:
        """
        Summarize the results in a DataFrame.
        Returns:
            pd.DataFrame: DataFrame with columns:
            ['id', 'model', 'config', 'r2_score', 'rmse_score'].
        TODO: include test data
        """
        data = {
            "id": [result.id for result in self.results],
            # TODO: if the id is changed to not include the model name,
            # we can include the model name in the DataFrame.
            # "model": [result.model.__class__.__name__ for result in self.results],
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

    # def plot_compare(self, ...): ...

    # def get_best_model(self, ...) -> Emulator:
    #     return self.get_best_result["model"]

    # def get_best_result(self, metric: str = "r2", ) -> Result: ...
