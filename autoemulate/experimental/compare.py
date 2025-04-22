from typing import Any, Optional

import numpy as np
import torchmetrics

from autoemulate.experimental.data.utils import InputTypeMixin
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.emulators.gaussian_process.exact import (
    GaussianProcessExact,
)
from autoemulate.experimental.model_selection import evaluate
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import InputLike


class AutoEmulate(InputTypeMixin):
    def __init__(
        self,
        x: InputLike,
        y: InputLike,
        models: Optional[list[type[Emulator]]] = None,
    ):
        if models is None:
            models = [GaussianProcessExact]
        self.models = models
        self.train_val, self.test = self._random_split(self._convert_to_dataset(x, y))

    def compare(self, n_iter: int = 10) -> dict[str, dict[str, Any]]:
        tuner = Tuner(self.train_val, y=None, n_iter=n_iter)
        models_evaluated = {}
        for model_cls in self.models:
            scores, configs = tuner.run(model_cls)
            best_score_idx = scores.index(max(scores))
            best_model_config = configs[best_score_idx]

            # refit model on all train+val data using the best config
            x, y = self._convert_to_tensors(self.train_val, y=None)
            model = model_cls(x, y, **best_model_config)
            model.fit(self.train_val, y=None)

            # predict on test data
            x_test, y_test = self._convert_to_tensors(self.test)
            y_test_pred = model.predict(x_test)
            r2_score = evaluate(y_test, y_test_pred, metric=torchmetrics.R2Score)
            rmse_score = np.sqrt(
                evaluate(y_test, y_test_pred, metric=torchmetrics.MeanSquaredError)
            )
            models_evaluated[model_cls.__name__] = {
                "config": best_model_config,
                "r2_score": r2_score,
                "rmse_score": rmse_score,
            }
            print(
                f"Model: {model_cls.__name__}, "
                f"Best params: {best_model_config}, "
                f"R2 score: {r2_score:.3f}, "
                f"RMSE score: {rmse_score:.3f}"
            )
        return models_evaluated
