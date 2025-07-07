from typing import Any

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

    # Equivalent of summarize_cv but with test data included
    # def summarize(self, ...) -> pd.DataFrame: ...

    # def plot_compare(self, ...): ...

    # def get_best_model(self, ...) -> Emulator:
    #     return self.get_best_result["model"]

    # def get_best_result(self, metric: str = "r2", ) -> Result: ...
