import logging
import warnings

import matplotlib.pyplot as plt
import torchmetrics

from autoemulate.experimental.data.utils import ConversionMixin, set_random_seed
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators import ALL_EMULATORS
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.model_selection import evaluate
from autoemulate.experimental.results import Result, Results
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import DeviceLike, InputLike


class AutoEmulate(ConversionMixin, TorchDeviceMixin, Results):
    def __init__(
        self,
        x: InputLike,
        y: InputLike,
        models: list[type[Emulator]] | None = None,
        device: DeviceLike | None = None,
        random_seed: int | None = None,
    ):
        Results.__init__(self)
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

        # Run the compare method with the provided models
        if not self.models:
            msg = (
                "No models provided or available for comparison. "
                "Please provide a list of models to compare."
            )
            raise ValueError(msg)
        self.compare()

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

    def log_compare(self, model_cls, best_config_for_this_model, r2_score, rmse_score):
        logger = logging.getLogger(__name__)
        msg = (
            f"Model: {model_cls.__name__}, "
            f"Best params: {best_config_for_this_model}, "
            f"R2 score: {r2_score:.3f}, "
            f"RMSE score: {rmse_score:.3f}"
        )
        logger.info(msg)

    def compare(
        self,
        n_iter: int = 10,
    ):
        tuner = Tuner(self.train_val, y=None, n_iter=n_iter, device=self.device)
        for id_num, model_cls in enumerate(self.models):
            scores, configs = tuner.run(model_cls)
            best_score_idx = scores.index(max(scores))
            best_config_for_this_model = configs[best_score_idx]

            val_x, val_y = self._convert_to_tensors(self.train_val)
            test_x, test_y = self._convert_to_tensors(self.test)
            m = model_cls(
                val_x, val_y, device=self.device, **best_config_for_this_model
            )
            m.fit(val_x, val_y)
            y_pred = m.predict(test_x)
            r2_score = evaluate(test_y, y_pred, torchmetrics.R2Score, self.device)
            rmse_score = evaluate(
                test_y, y_pred, torchmetrics.MeanSquaredError, self.device
            )
            result = Result(
                id=model_cls.__name__ + str(id_num + 1),
                model=m,
                config=best_config_for_this_model,
                r2_score=r2_score,
                rmse_score=rmse_score,
            )
            self.add_result(result)
            self.log_compare(
                model_cls, best_config_for_this_model, r2_score, rmse_score
            )

    def plot(
        self,
        result_id: str,
        # input_index: list[int] | None = None
    ):
        """
        Plot the evaluation of the model with the given result_id.
        Parameters
        ----------
        result_id: str
            The ID of the model to plot.
        input_index: list[int] | None
            The indices of the inputs to plot. If None, all inputs are plotted.
        """
        if result_id not in self._id_to_result:
            raise ValueError(f"No result found with ID: {result_id}")

        test_x, test_y = self._convert_to_tensors(self.test)
        model = self.get_result(result_id).model

        # Re-run prediction with just this model to get the predictions
        y_pred = model.predict(test_x)
        r2_score = evaluate(test_y, y_pred, torchmetrics.R2Score, self.device)
        rmse_score = evaluate(
            test_y, y_pred, torchmetrics.MeanSquaredError, self.device
        )

        # Plot the evaluation
        # TODO: test plot - reimplement with the equivalent of plot_eval in v0
        plt.figure(figsize=(10, 6))
        plt.scatter(test_y, y_pred, alpha=0.5)  # type: ignore PGH003
        plt.plot(
            [test_y.min(), test_y.max()],
            [test_y.min(), test_y.max()],
            color="red",
            linestyle="--",
        )
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        t = f"Model: {result_id} - R2: {r2_score:.3f}, RMSE: {rmse_score:.3f}"
        plt.title(t)
        plt.grid()
        plt.show()
