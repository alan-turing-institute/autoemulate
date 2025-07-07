import warnings
from typing import Any

import numpy as np
import tqdm
from sklearn.model_selection import BaseCrossValidator, KFold

from autoemulate.experimental.data.utils import ConversionMixin, set_random_seed
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators import ALL_EMULATORS
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.logging_config import configure_logging
from autoemulate.experimental.model_selection import cross_validate
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import DeviceLike, InputLike


class AutoEmulate(ConversionMixin, TorchDeviceMixin):
    def __init__(  # noqa: PLR0913
        self,
        x: InputLike,
        y: InputLike,
        models: list[type[Emulator]] | None = None,
        device: DeviceLike | None = None,
        random_seed: int | None = None,
        log_level: str = "info",
        progress_bar: bool = False,
    ):
        self.random_seed = random_seed
        TorchDeviceMixin.__init__(self, device=device)
        # TODO: refactor in https://github.com/alan-turing-institute/autoemulate/issues/400
        x, y = self._convert_to_tensors(x, y)
        x, y = self._move_tensors_to_device(x, y)

        # Set default models if None
        updated_models = self.get_models(models)

        # Filter models to only be those that can handle multioutput data
        if y.shape[1] > 1:
            updated_models = self.filter_models_if_multioutput(
                updated_models, models is not None
            )

        self.models = updated_models
        if random_seed is not None:
            set_random_seed(seed=random_seed)
        self.train_val, self.test = self._random_split(self._convert_to_dataset(x, y))

        self.logger = configure_logging(level=log_level)
        self.progress_bar = progress_bar

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

    def log_compare(self, best_model_name, best_model_config, r2_score, rmse_score):
        msg = (
            "Comparison results:\n"
            f"Best Model: {best_model_name}, "
            f"Best params: {best_model_config}, "
            f"R2 score: {r2_score:.3f}, "
            f"RMSE score: {rmse_score:.3f}"
        )
        self.logger.debug(msg)

    def compare(
        self,
        n_iter: int = 10,
        cv: type[BaseCrossValidator] = KFold,
        cv_seed: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        tuner = Tuner(self.train_val, y=None, n_iter=n_iter, device=self.device)
        models_evaluated = {}

        self.logger.info(
            "Comparing %s", [model_cls.__name__ for model_cls in self.models]
        )
        for idx, model_cls in tqdm.tqdm(
            enumerate(self.models), disable=not self.progress_bar
        ):
            self.logger.info(
                "Running Model: %s: %d/%d",
                model_cls.__name__,
                idx + 1,
                len(self.models),
            )

            self.logger.debug('Running tuner for model "%s"', model_cls.__name__)
            scores, configs = tuner.run(model_cls)
            best_score_idx = scores.index(max(scores))
            best_model_config = configs[best_score_idx]
            self.logger.debug(
                'Tuner found best config for model "%s": %s with score: %s',
                model_cls.__name__,
                best_model_config,
                scores[best_score_idx],
            )

            self.logger.debug(
                'Running cross-validation for model "%s" for "%s" iterations',
                model_cls.__name__,
                n_iter,
            )
            cv_results = cross_validate(
                cv=cv(
                    random_state=cv_seed  # type: ignore PGH003
                ),
                dataset=self.train_val.dataset,
                model=model_cls,
                random_seed=None,  # Does not need to be set the same for each model
                **best_model_config,
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
            self.logger.debug(
                'Cross-validation for model "%s"'
                " completed with R2 score: %.3f, RMSE score: %.3f",
                model_cls.__name__,
                r2_score,
                rmse_score,
            )
            self.logger.info("Finished running Model: %s\n", model_cls.__name__)

        # Find the best model based on R2 score
        best_model_cls = max(
            models_evaluated.items(),
            key=lambda item: item[1]["r2_score"],
        )[0]
        best_model_config = models_evaluated[best_model_cls]["config"]
        r2_score = models_evaluated[best_model_cls]["r2_score"]
        rmse_score = models_evaluated[best_model_cls]["rmse_score"]

        self.log_compare(best_model_cls, best_model_config, r2_score, rmse_score)
        self.logger.info("Compare completed successfully.")
        return models_evaluated


if __name__ == "__main__":
    x = np.random.rand(100, 3)
    y = np.random.rand(100, 2)

    autoemulate = AutoEmulate(x, y, log_level="debug")

    results = autoemulate.compare(n_iter=5)
