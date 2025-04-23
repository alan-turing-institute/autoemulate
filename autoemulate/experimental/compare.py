import logging
from typing import Any

import numpy as np
from sklearn.model_selection import BaseCrossValidator, KFold

from autoemulate.experimental.data.utils import InputTypeMixin
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.emulators.gaussian_process.exact import (
    GaussianProcessExact,
)
from autoemulate.experimental.model_selection import cross_validate
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import InputLike


class AutoEmulate(InputTypeMixin):
    def __init__(
        self,
        x: InputLike,
        y: InputLike,
        models: list[type[Emulator]] | None = None,
    ):
        if models is None:
            models = [GaussianProcessExact]
        self.models = models
        self.train_val, self.test = self._random_split(self._convert_to_dataset(x, y))

    def log_compare(self, model_cls, best_model_config, r2_score, rmse_score):
        logger = logging.getLogger(__name__)
        msg = (
            f"Model: {model_cls.__name__}, "
            f"Best params: {best_model_config}, "
            f"R2 score: {r2_score:.3f}, "
            f"RMSE score: {rmse_score:.3f}"
        )
        logger.info(msg)

    def compare(
        self, n_iter: int = 10, cv: type[BaseCrossValidator] = KFold
    ) -> dict[str, dict[str, Any]]:
        tuner = Tuner(self.train_val, y=None, n_iter=n_iter)
        models_evaluated = {}
        for model_cls in self.models:
            scores, configs = tuner.run(model_cls)
            best_score_idx = scores.index(max(scores))
            best_model_config = configs[best_score_idx]
            cv_results = cross_validate(
                cv(), self.train_val.dataset, model_cls, **best_model_config
            )
            r2_score, rmse_score = (
                np.mean(cv_results["r2"]),
                np.mean(cv_results["rmse"]),
            )
            models_evaluated[model_cls.__name__] = {
                "config": best_model_config,
                "r2_score": r2_score,
                "rmse_score": rmse_score,
            }
            self.log_compare(model_cls, best_model_config, r2_score, rmse_score)
        return models_evaluated
