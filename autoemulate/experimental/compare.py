import logging
import warnings
from typing import Any

import numpy as np
from sklearn.model_selection import BaseCrossValidator, KFold

from autoemulate.experimental.data.utils import ConversionMixin
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators import ALL_EMULATORS
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.model_selection import cross_validate
from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import DeviceLike, InputLike


class AutoEmulate(ConversionMixin, TorchDeviceMixin):
    def __init__(  # noqa: PLR0913
        self,
        x: InputLike,
        y: InputLike,
        models: list[type[Emulator]] | None = None,
        x_transforms_list: list[list[AutoEmulateTransform]] | None = None,
        y_transforms_list: list[list[AutoEmulateTransform]] | None = None,
        device: DeviceLike | None = None,
    ):
        TorchDeviceMixin.__init__(self, device=device)
        # TODO: refactor in https://github.com/alan-turing-institute/autoemulate/issues/400
        x, y = self._convert_to_tensors(x, y)
        x, y = self._move_tensors_to_device(x, y)

        # Transforms to search over
        self.x_transforms_list = x_transforms_list or [[]]
        self.y_transforms_list = y_transforms_list or [[]]

        # Set default models if None
        updated_models = self.get_models(models)

        # Filter models to only be those that can handle multioutput data
        if y.shape[1] > 1:
            updated_models = self.filter_models_if_multioutput(
                updated_models, models is not None
            )

        self.models = updated_models
        self.train_val, self.test = self._random_split(self._convert_to_dataset(x, y))

    @staticmethod
    def all_emulators() -> list[type[Emulator]]:
        return ALL_EMULATORS

    def get_models(self, models: list[type[Emulator]] | None) -> list[type[Emulator]]:
        if models is None:
            return self.all_emulators()
        return models

    def filter_models_if_multioutput(
        self, models: list[type[Emulator]], warn: bool
    ) -> list[type[Emulator]]:
        updated_models = []
        for model in models:
            if not model.is_multioutput():
                if warn:
                    msg = (
                        f"Model ({model}) is not multioutput but the data is "
                        f"multioutput. Skipping model ({model})..."
                    )
                    warnings.warn(msg, stacklevel=2)
            else:
                updated_models.append(model)
        return updated_models

    def log_compare(  # noqa: PLR0913
        self,
        model_cls,
        x_transforms,
        y_transforms,
        best_model_config,
        r2_score,
        rmse_score,
    ):
        logger = logging.getLogger(__name__)
        msg = (
            f"Model: {model_cls.__name__}, "
            f"x transforms: {x_transforms}, "
            f"y transforms: {y_transforms}",
            f"Best params: {best_model_config}, "
            f"R2 score: {r2_score:.3f}, "
            f"RMSE score: {rmse_score:.3f}",
        )
        logger.info(msg)

    def compare(
        self, n_iter: int = 10, cv: type[BaseCrossValidator] = KFold
    ) -> dict[str, dict[str, Any]]:
        tuner = Tuner(self.train_val, y=None, n_iter=n_iter, device=self.device)
        models_evaluated = {}
        for x_transforms in self.x_transforms_list:
            for y_transforms in self.y_transforms_list:
                for model_cls in self.models:
                    scores, configs = tuner.run(model_cls, x_transforms, y_transforms)
                    best_score_idx = scores.index(max(scores))
                    best_model_config = configs[best_score_idx]
                    cv_results = cross_validate(
                        cv(), self.train_val.dataset, model_cls, **best_model_config
                    )
                    r2_score, rmse_score = (
                        np.mean(cv_results["r2"]),
                        np.mean(cv_results["rmse"]),
                    )
                    models_evaluated[
                        (
                            # TODO: refactor names of transforms and target_transforms
                            "-".join(str(t) for t in x_transforms),
                            "-".join(str(t) for t in y_transforms),
                            model_cls.model_name(),
                        )
                    ] = {
                        "config": best_model_config,
                        "r2_score": r2_score,
                        "rmse_score": rmse_score,
                    }
                    self.log_compare(
                        model_cls,
                        x_transforms,
                        y_transforms,
                        best_model_config,
                        r2_score,
                        rmse_score,
                    )
        return models_evaluated
