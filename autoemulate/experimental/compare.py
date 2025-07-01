import logging
import warnings
from typing import Any

import numpy as np
from sklearn.model_selection import BaseCrossValidator, KFold

from autoemulate.experimental.data.utils import ConversionMixin, set_random_seed
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators import ALL_EMULATORS
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.model_selection import cross_validate
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import DeviceLike, InputLike


class AutoEmulate(ConversionMixin, TorchDeviceMixin):
    def __init__(
        self,
        x: InputLike,
        y: InputLike,
        models: list[type[Emulator]] | None = None,
        device: DeviceLike | None = None,
        random_seed: int | None = None,
    ):
        self.random_seed = random_seed
        TorchDeviceMixin.__init__(self, device=device)
        # TODO: refactor in https://github.com/alan-turing-institute/autoemulate/issues/400
        x, y = self._convert_to_tensors(x, y)
        self.x, self.y = self._move_tensors_to_device(x, y)

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

        self.train_val, self.test = self._random_split(
            self._convert_to_dataset(self.x, self.y)
        )
        self._comparison_results: dict[str, dict[str, Any]] | None = None

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
        self,
        n_iter: int = 10,
        cv: type[BaseCrossValidator] = KFold,
        cv_seed: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        tuner = Tuner(self.train_val, y=None, n_iter=n_iter, device=self.device)
        models_evaluated = {}
        for model_cls in self.models:
            scores, configs = tuner.run(model_cls)
            best_score_idx = scores.index(max(scores))
            best_model_config = configs[best_score_idx]
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
            self.log_compare(model_cls, best_model_config, r2_score, rmse_score)

        # Store comparison results for refit method
        self._comparison_results = models_evaluated
        return models_evaluated

    def refit(
        self,
        model_name: str | None = None,
        metric: str = "r2_score",
        x: InputLike | None = None,
        y: InputLike | None = None,
    ) -> Emulator:
        """
        Refit the best model from comparison results on all available data.

        This method can only be called after `compare()` has been run. By default,
        it selects the model with the highest R² score from the comparison results.
        Alternatively, a specific model name can be provided.

        Parameters
        ----------
        model_name : str | None, optional
            Name of the model to refit. If None, selects the best model based on
            the specified metric from comparison results. Must be a model name that
            was included in the comparison.
        metric : str, optional
            Metric to use for selecting the best model. Default is 'r2_score'.
        x : InputLike | None, optional
            Input data to use for refitting. If None, uses the original data
            passed to AutoEmulate constructor.
        y : InputLike | None, optional
            Target data to use for refitting. If None, uses the original data
            passed to AutoEmulate constructor.

        Returns
        -------
        Emulator
            The fitted emulator trained on the specified data.

        Raises
        ------
        RuntimeError
            If `compare()` has not been run yet.
        ValueError
            If the specified model_name was not found in comparison results.
        """
        if self._comparison_results is None:
            msg = (
                "Must run compare() before calling refit(). "
                "No comparison results found."
            )
            raise RuntimeError(msg)

        # Select model to refit
        if model_name is None:
            # Find model with best R² score
            best_model_name = max(
                self._comparison_results.keys(),
                key=lambda k: self._comparison_results[k][metric],  # type: ignore[index]
            )
        else:
            if model_name not in self._comparison_results:
                available_models = list(self._comparison_results.keys())
                raise ValueError(
                    f"Model '{model_name}' not found in comparison results. "
                    f"Available models: {available_models}"
                )
            best_model_name = model_name

        # Get model class and config
        model_cls = next(
            model for model in self.models if model.__name__ == best_model_name
        )
        best_config = self._comparison_results[best_model_name]["config"]

        # Use provided data or default to original data
        if x is None and y is None:
            refit_x, refit_y = self.x, self.y
        elif x is not None and y is not None:
            # Convert both x and y to tensors
            refit_x, refit_y = self._convert_to_tensors(x, y)
            refit_x, refit_y = self._move_tensors_to_device(refit_x, refit_y)
        elif x is not None and y is None:
            # Convert only x, use original y
            refit_x, _ = self._convert_to_tensors(x, self.y)
            refit_x, _ = self._move_tensors_to_device(refit_x, self.y)
            refit_y = self.y
        else:  # x is None and y is not None
            # Convert only y, use original x
            _, refit_y = self._convert_to_tensors(self.x, y)
            _, refit_y = self._move_tensors_to_device(self.x, refit_y)
            refit_x = self.x

        # Create and fit model with best configuration
        if self.random_seed is not None:
            set_random_seed(seed=self.random_seed)

        model = model_cls(x=refit_x, y=refit_y, device=self.device, **best_config)
        model.fit(refit_x, refit_y)

        return model
