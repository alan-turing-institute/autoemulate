import logging
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import torchmetrics

from autoemulate.experimental.data.utils import ConversionMixin, set_random_seed
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators import ALL_EMULATORS
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.emulators.transformed.base import TransformedEmulator
from autoemulate.experimental.model_selection import evaluate
from autoemulate.experimental.plotting import (
    calculate_subplot_layout,
    display_figure,
    plot_Xy,
)
from autoemulate.experimental.results import Result, Results
from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.transforms.standardize import StandardizeTransform
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import DeviceLike, DistributionLike, InputLike


class AutoEmulate(ConversionMixin, TorchDeviceMixin, Results):
    def __init__(  # noqa: PLR0913
        self,
        x: InputLike,
        y: InputLike,
        models: list[type[Emulator]] | None = None,
        x_transforms_list: list[list[AutoEmulateTransform]] | None = None,
        y_transforms_list: list[list[AutoEmulateTransform]] | None = None,
        n_iter: int = 10,
        n_splits: int = 5,
        shuffle: bool = True,
        device: DeviceLike | None = None,
        random_seed: int | None = None,
    ):
        Results.__init__(self)
        self.random_seed = random_seed
        TorchDeviceMixin.__init__(self, device=device)
        x, y = self._convert_to_tensors(x, y)
        x, y = self._move_tensors_to_device(x, y)

        # Transforms to search over
        self.x_transforms_list = x_transforms_list or [[StandardizeTransform()]]
        self.y_transforms_list = y_transforms_list or [[StandardizeTransform()]]

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

        # Assign tuner parameters
        self.n_splits = n_splits
        self.shuffle = shuffle

        # Run compare
        self.compare(n_iter=n_iter)

    @staticmethod
    def all_emulators() -> list[type[Emulator]]:
        return ALL_EMULATORS

    @staticmethod
    def list_emulators() -> pd.DataFrame:
        """
        Return a dataframe with the model_name and short_name
        of all available emulators.
        Returns:
            pd.DataFrame: DataFrame with columns ['model_name', 'short_name'].
        """
        return pd.DataFrame(
            {
                "model_name": [emulator.model_name() for emulator in ALL_EMULATORS],
                "short_name": [emulator.short_name() for emulator in ALL_EMULATORS],
            }
        )

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
        best_config_for_this_model,
        r2_score,
        rmse_score,
    ):
        logger = logging.getLogger(__name__)
        msg = (
            f"Model: {model_cls.__name__}, "
            f"x transforms: {x_transforms}, "
            f"y transforms: {y_transforms}",
            f"Best params: {best_config_for_this_model}, "
            f"R2 score: {r2_score:.3f}, "
            f"RMSE score: {rmse_score:.3f}",
        )
        logger.info(msg)

    def compare(self, n_iter: int = 100):
        tuner = Tuner(self.train_val, y=None, n_iter=n_iter, device=self.device)
        for x_transforms in self.x_transforms_list:
            for y_transforms in self.y_transforms_list:
                for id_num, model_cls in enumerate(self.models):
                    scores, configs = tuner.run(
                        model_cls,
                        x_transforms,
                        y_transforms,
                        n_splits=self.n_splits,
                        shuffle=self.shuffle,
                    )
                    best_score_idx = scores.index(max(scores))
                    best_config_for_this_model = configs[best_score_idx]
                    train_val_x, train_val_y = self._convert_to_tensors(self.train_val)
                    test_x, test_y = self._convert_to_tensors(self.test)
                    transformed_emulator = TransformedEmulator(
                        train_val_x,
                        train_val_y,
                        model=model_cls,
                        x_transforms=x_transforms,
                        y_transforms=y_transforms,
                        device=self.device,
                        **best_config_for_this_model,
                    )
                    transformed_emulator.fit(train_val_x, train_val_y)
                    y_pred = transformed_emulator.predict(test_x)
                    r2_score = evaluate(
                        test_y, y_pred, torchmetrics.R2Score, self.device
                    )
                    rmse_score = evaluate(
                        test_y, y_pred, torchmetrics.MeanSquaredError, self.device
                    )
                    result = Result(
                        id=model_cls.short_name() + str(id_num + 1),
                        model_name=model_cls.model_name(),
                        model=transformed_emulator,
                        config=best_config_for_this_model,
                        r2_score=r2_score,
                        rmse_score=rmse_score,
                    )
                    self.add_result(result)
                    self.log_compare(
                        model_cls,
                        best_config_for_this_model,
                        x_transforms,
                        y_transforms,
                        r2_score,
                        rmse_score,
                    )

    def plot(  # noqa: PLR0912, PLR0915
        self,
        result_id: str,
        input_index: list[int] | int | None = None,
        output_index: list[int] | int | None = None,
        figsize=None,
        n_cols: int = 3,
    ):
        """
        Plot the evaluation of the model with the given result_id.
        Parameters
        ----------
        result_id: str
            The ID of the model to plot.
        input_index: int
            The index of the input feature to plot against the output.
        output_index: int
            The index of the output feature to plot against the input.
        """
        if result_id not in self._id_to_result:
            raise ValueError(f"No result found with ID: {result_id}")

        test_x, test_y = self._convert_to_tensors(self.test)
        model = self.get_result(result_id).model

        # Re-run prediction with just this model to get the predictions
        y_pred = model.predict(test_x)
        y_variance = None
        if isinstance(y_pred, DistributionLike):
            y_variance = y_pred.variance
            y_pred = y_pred.mean
        r2_score = evaluate(test_y, y_pred, torchmetrics.R2Score, self.device)

        # Convert to numpy arrays for plotting and ensure correct shapes
        test_x, test_y = self._convert_to_numpy(test_x, test_y)
        y_pred, _ = self._convert_to_numpy(y_pred, None)
        assert test_x is not None
        assert test_y is not None
        assert y_pred is not None
        test_x = self._ensure_numpy_2d(test_x)
        test_y = self._ensure_numpy_2d(test_y)
        y_pred = self._ensure_numpy_2d(y_pred)
        if y_variance is not None:
            y_variance, _ = self._convert_to_numpy(y_variance, None)
            y_variance = self._ensure_numpy_2d(y_variance)

        _, n_features = test_x.shape

        n_outputs = test_y.shape[1] if test_y.ndim > 1 else 1

        # Handle input and output indices
        if input_index is None:
            input_index = list(range(n_features))
        elif isinstance(input_index, int):
            input_index = [input_index]

        if output_index is None:
            output_index = list(range(n_outputs))
        elif isinstance(output_index, int):
            output_index = [output_index]

        # check that input_index and output_index are valid
        if any(idx >= n_features for idx in input_index):
            msg = f"input_index {input_index} is out of range. "
            msg += f"The index should be between 0 and {n_features - 1}."
            raise ValueError(msg)
        if any(idx >= n_outputs for idx in output_index):
            msg = f"output_index {output_index} is out of range. "
            msg += f"The index should be between 0 and {n_outputs - 1}."
            raise ValueError(msg)

        # Calculate number of subplots
        n_plots = len(input_index) * len(output_index)

        # Calculate number of rows
        n_rows, n_cols = calculate_subplot_layout(n_plots, n_cols)

        # Set up the figure
        if figsize is None:
            figsize = (5 * n_cols, 4 * n_rows)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axs = axs.flatten()

        plot_index = 0
        for out_idx in output_index:
            for in_idx in input_index:
                if plot_index < len(axs):
                    plot_Xy(
                        test_x[:, in_idx],
                        test_y[:, out_idx],
                        y_pred[:, out_idx],
                        y_variance[:, out_idx] if y_variance is not None else None,
                        ax=axs[plot_index],
                        title=f"$X_{in_idx}$ vs. $y_{out_idx}$",
                        input_index=in_idx,
                        output_index=out_idx,
                        r2_score=r2_score,
                    )
                    plot_index += 1

        # Hide any unused subplots
        for ax in axs[plot_index:]:
            ax.set_visible(False)
        plt.tight_layout()

        return display_figure(fig)
