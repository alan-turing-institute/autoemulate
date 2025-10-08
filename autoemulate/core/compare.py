import inspect
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from torch.distributions import Transform

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.logging_config import get_configured_logger
from autoemulate.core.model_selection import bootstrap, evaluate, r2_metric
from autoemulate.core.plotting import (
    calculate_subplot_layout,
    create_and_plot_slice,
    display_figure,
    plot_xy,
)
from autoemulate.core.reinitialize import fit_from_reinitialized
from autoemulate.core.results import Result, Results
from autoemulate.core.save import ModelSerialiser
from autoemulate.core.tuner import Tuner
from autoemulate.core.types import (
    DeviceLike,
    InputLike,
    ModelParams,
    TransformedEmulatorParams,
)
from autoemulate.data.utils import ConversionMixin, set_random_seed
from autoemulate.emulators import (
    ALL_EMULATORS,
    DEFAULT_EMULATORS,
    PYTORCH_EMULATORS,
    get_emulator_class,
)
from autoemulate.emulators.base import Emulator
from autoemulate.emulators.transformed.base import TransformedEmulator
from autoemulate.transforms.base import AutoEmulateTransform
from autoemulate.transforms.standardize import StandardizeTransform


class AutoEmulate(ConversionMixin, TorchDeviceMixin, Results):
    """
    Automated emulator fitting.

    The AutoEmulate class is the main class of the AutoEmulate package.
    It is used to set up and compare different emulator models on a given dataset.
    It can also be used to summarise and visualise results, and to save and load models.

    """

    def __init__(
        self,
        x: InputLike,
        y: InputLike,
        models: list[type[Emulator] | str] | None = None,
        x_transforms_list: list[list[Transform | dict]] | None = None,
        y_transforms_list: list[list[Transform | dict]] | None = None,
        model_params: None | ModelParams | dict = None,
        transformed_emulator_params: None | TransformedEmulatorParams = None,
        only_probabilistic: bool = False,
        n_iter: int = 10,
        n_splits: int = 5,
        shuffle: bool = True,
        n_bootstraps: int | None = 100,
        max_retries: int = 3,
        device: DeviceLike | None = None,
        random_seed: int | None = None,
        log_level: str = "progress_bar",
    ):
        """
        Initialize the AutoEmulate class.

        Parameters
        ----------
        x: InputLike
            Input features.
        y: InputLike or None
            Target values (not needed if x is a Dataset).
        models: list[type[Emulator]] | None
            List of emulator classes to compare. If None, all available emulators
            are used.
        x_transforms_list: list[list[Transform]] | None
            An optional list of sequences of transforms to apply to the input data.
            Defaults to None, in which case the data is standardized.
        y_transforms_list: list[list[Transform]] | None
            An optional list of sequences of transforms to apply to the output data.
            Defaults to None, in which case the data is standardized.
        model_params: ModelParams | None
            If None, default behaviour, the model hyperparameters are tuned using
            cross-validation. If provided, the model hyperparameters are set to the
            provided values and no tuning is performed. If an empty dictionary {} is
            provided the default parameters for each model are used without tuning.
        transformed_emulator_params: None | TransformedEmulatorParams
            Parameters for the transformed emulator. Defaults to None.
        only_probabilistic: bool
            If True, only probabilistic emulators are used. Defaults to False.
        n_iter: int
            Number of parameter settings to randomly sample and test during tuning.
        n_splits: int
            Number of cross validation folds to split data into. Defaults to 5.
        shuffle: bool
            Whether to shuffle data before splitting into cross validation folds.
            Defaults to True.
        n_bootstraps: int | None
            Number of times to resample the data when evaluating performance. When None
            the evaluation uses single test set and returns a single value with no
            measure of the uncertainty in test performance. Defaults to 100.
        device: DeviceLike | None
            Device to run the emulators. If None, uses the default device (usually CPU
            or GPU). Defaults to None.
        random_seed: int | None
            Random seed for reproducibility. If None, no seed is set. Defaults to None.
        log_level: str
            Logging level. Can be "progress_bar", "debug", "info", "warning",
            "error", or "critical". Defaults to "progress_bar". If "progress_bar",
            it will show a progress bar during model comparison. It will set the
            logging level to "error" to avoid cluttering the output
            with debug/info logs.
        """
        Results.__init__(self)
        self.random_seed = random_seed
        TorchDeviceMixin.__init__(self, device=device)
        x, y = self._convert_to_tensors(x, y)
        x, y = self._move_tensors_to_device(x, y)

        # Transforms to search over
        self.x_transforms_list = [
            self.get_transforms(transforms)
            for transforms in (x_transforms_list or [[StandardizeTransform()]])
        ]
        self.y_transforms_list = [
            self.get_transforms(transforms)
            for transforms in (y_transforms_list or [[StandardizeTransform()]])
        ]

        # Set default models if None
        updated_models = self.get_models(models, only_probabilistic=only_probabilistic)

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
        self.n_iter = n_iter
        self.n_bootstraps = n_bootstraps
        self.max_retries = max_retries
        self.model_params = model_params or {}
        self.model_tuning = model_params is None
        self.transformed_emulator_params = transformed_emulator_params or {}

        # Set up logger and ModelSerialiser for saving models
        self.logger, self.progress_bar = get_configured_logger(log_level)
        self.model_serialiser = ModelSerialiser(self.logger)

        # Run compare
        self.compare()

    @staticmethod
    def all_emulators() -> list[type[Emulator]]:
        """Return a list of all available emulators."""
        return ALL_EMULATORS

    @staticmethod
    def default_emulators() -> list[type[Emulator]]:
        """Return a list of default emulators used by AutoEmulate."""
        return DEFAULT_EMULATORS

    @staticmethod
    def pytorch_emulators() -> list[type[Emulator]]:
        """Return a list of all available PyTorch emulators."""
        return PYTORCH_EMULATORS

    @staticmethod
    def probablistic_emulators() -> list[type[Emulator]]:
        """Return a list of all available probabilistic emulators."""
        return [emulator for emulator in ALL_EMULATORS if emulator.supports_uq]

    @staticmethod
    def list_emulators(default_only: bool = True) -> pd.DataFrame:
        """Return a dataframe with info on all available emulators.

        The dataframe includes the model name and whether it has a PyTorch backend (and
        autodiff), supports multioutput data and provides uncertainty quantification.

        Parameters
        ----------
        subset: bool
            Whether to display only default or all available emulators. Defaults to
            True (default emulators only).


        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
                - 'Emulator',
                - 'PyTorch',
                - 'Multioutput',
                - 'Uncertainty_Quantification',
                - 'Automatic_Differentiation`
        """
        emulator_set = DEFAULT_EMULATORS if default_only else ALL_EMULATORS
        return pd.DataFrame(
            {
                "Emulator": [emulator.model_name() for emulator in emulator_set],
                "PyTorch": [
                    emulator in AutoEmulate.pytorch_emulators()
                    for emulator in emulator_set
                ],
                "Multioutput": [emulator.is_multioutput() for emulator in emulator_set],
                "Uncertainty_Quantification": [
                    emulator in AutoEmulate.probablistic_emulators()
                    for emulator in emulator_set
                ],
                "Automatic_Differentiation": [
                    emulator in AutoEmulate.pytorch_emulators()
                    for emulator in emulator_set
                ],
                # TODO: short_name not currently used for anything, so commented out
                # "short_name": [emulator.short_name() for emulator in ALL_EMULATORS],
            }
        )

    def get_models(
        self,
        models: list[type[Emulator] | str] | None = None,
        only_probabilistic: bool = False,
    ) -> list[type[Emulator]]:
        """
        Return a list of the model classes for comparisons.

        Parameters
        ----------
        models: list[type[Emulator] | str] | None
            List of model classes or names to use for comparison. If None, all available
            emulators are used (or subset based on only_pytorch and only_probabilistic).
        only_pytorch: bool
            If True, only PyTorch emulators are returned. Defaults to False.
        only_probabilistic: bool
            If True, only probabilistic emulators are returned. Defaults to False.
        """
        if models is None:
            if only_probabilistic:
                return [
                    emulator
                    for emulator in self.default_emulators()
                    if emulator in self.probablistic_emulators()
                ]
            return self.default_emulators()

        model_classes = []
        for model in models:
            if isinstance(model, str):
                model_classes.append(get_emulator_class(model))
            elif issubclass(model, Emulator):
                model_classes.append(model)
            else:
                raise ValueError(
                    f"Invalid model type: {type(model)}. "
                    "Expected a string or a subclass of Emulator"
                )
        return model_classes

    def get_transforms(
        self, transforms: list[Transform | dict[str, object]]
    ) -> list[Transform]:
        """Process and return a list of transforms."""
        processed_transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                deserialized_transform = AutoEmulateTransform.from_dict(transform)
                processed_transforms.append(deserialized_transform)
            elif isinstance(transform, Transform):
                processed_transforms.append(transform)
            else:
                msg = f"Invalid transform type: {type(transform)}"
                raise ValueError(msg)
        return processed_transforms

    def filter_models_if_multioutput(
        self, models: list[type[Emulator]], warn: bool
    ) -> list[type[Emulator]]:
        """Filter models to only include those that support multi-output data."""
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

    def log_compare(
        self,
        best_model_name,
        x_transforms,
        y_transforms,
        best_params_for_this_model,
        r2_score,
        rmse_score,
    ):
        """Log the comparison results."""
        msg = (
            "Comparison results:\n"
            f"Best Model: {best_model_name}, "
            f"x transforms: {x_transforms}, "
            f"y transforms: {y_transforms}",
            f"Best params: {best_params_for_this_model}, "
            f"R2 score: {r2_score:.3f}, "
            f"RMSE score: {rmse_score:.3f}",
        )
        self.logger.debug(msg)

    def compare(self):
        """
        Compare different models on the provided dataset.

        The method will:
        - Loop over all combinations of x and y transforms and models.
        - Set up the tuner with the training/validation data.
        - Tune hyperparameters for each model.
        - Fit the best model with the tuned hyperparameters.
        - Evaluate the performance of the best model on the test data.
        - Log the results.
        - Save the best model and its parameters.
        """
        tuner = Tuner(self.train_val, y=None, n_iter=self.n_iter, device=self.device)
        self.logger.info(
            "Comparing %s", [model_cls.__name__ for model_cls in self.models]
        )
        for x_transforms in self.x_transforms_list:
            for y_transforms in self.y_transforms_list:
                for id, model_cls in tqdm.tqdm(
                    enumerate(self.models),
                    disable=not self.progress_bar,
                    desc="Comparing models",
                    total=len(self.models),
                    unit="model",
                    unit_scale=True,
                ):
                    # Since refitting the model can fail after tuning, max_retries
                    # is used to retry the whole tuning and fitting process
                    for attempt in range(self.max_retries):
                        try:
                            self.logger.info(
                                "Running Model: %s: %d/%d (attempt %d/%d)",
                                model_cls.__name__,
                                id + 1,
                                len(self.models),
                                attempt + 1,
                                self.max_retries,
                            )
                            if self.model_tuning:
                                self.logger.debug(
                                    'Running tuner for model "%s"',
                                    model_cls.__name__,
                                )
                                scores, params_list = tuner.run(
                                    model_cls,
                                    x_transforms,
                                    y_transforms,
                                    n_splits=self.n_splits,
                                    shuffle=self.shuffle,
                                    transformed_emulator_params=self.transformed_emulator_params,
                                )
                                mean_scores = [
                                    np.mean(score).item() for score in scores
                                ]
                                best_score_idx = np.argmax(mean_scores)
                                best_params_for_this_model = params_list[best_score_idx]
                                self.logger.debug(
                                    'Tuner found best params for model "%s": '
                                    "%s with score: %.3f",
                                    model_cls.__name__,
                                    best_params_for_this_model,
                                    mean_scores[best_score_idx],
                                )
                            else:
                                self.logger.debug(
                                    'Skipping tuning for model "%s", using default'
                                    "parameters",
                                    model_cls.__name__,
                                )
                                # extract default parameters from the model's __init__
                                init_sig = inspect.signature(model_cls.__init__)
                                init_params = {
                                    param_name: param.default
                                    for param_name, param in init_sig.parameters.items()
                                    if param_name in model_cls.get_tune_params()
                                }
                                best_params_for_this_model = init_params

                            self.logger.debug(
                                'Running cross-validation for model "%s" '
                                'for "%s" iterations',
                                model_cls.__name__,
                                self.n_iter,
                            )
                            train_val_x, train_val_y = self._convert_to_tensors(
                                self.train_val
                            )
                            test_x, test_y = self._convert_to_tensors(self.test)
                            transformed_emulator = TransformedEmulator(
                                train_val_x,
                                train_val_y,
                                model=model_cls,
                                x_transforms=x_transforms,
                                y_transforms=y_transforms,
                                device=self.device,
                                **best_params_for_this_model,
                                **self.transformed_emulator_params,
                            )

                            # This can fail for some model params
                            transformed_emulator.fit(train_val_x, train_val_y)

                            (
                                (r2_train_val, r2_train_val_std),
                                (rmse_train_val, rmse_train_val_std),
                            ) = bootstrap(
                                transformed_emulator,
                                train_val_x,
                                train_val_y,
                                n_bootstraps=self.n_bootstraps,
                                device=self.device,
                            )
                            (r2_test, r2_test_std), (rmse_test, rmse_test_std) = (
                                bootstrap(
                                    transformed_emulator,
                                    test_x,
                                    test_y,
                                    n_bootstraps=self.n_bootstraps,
                                    device=self.device,
                                )
                            )

                            self.logger.debug(
                                'Cross-validation for model "%s"'
                                " completed with test mean (std) R2 score: %.3f (%.3f),"
                                " mean (std) RMSE score: %.3f (%.3f)",
                                model_cls.__name__,
                                r2_test,
                                r2_test_std,
                                rmse_test,
                                rmse_test_std,
                            )
                            self.logger.info(
                                "Finished running Model: %s\n", model_cls.__name__
                            )
                            result = Result(
                                id=id,
                                model_name=transformed_emulator.untransformed_model_name,
                                model=transformed_emulator,
                                params=best_params_for_this_model,
                                r2_test=r2_test,
                                rmse_test=rmse_test,
                                r2_test_std=r2_test_std,
                                rmse_test_std=rmse_test_std,
                                r2_train=r2_train_val,
                                rmse_train=rmse_train_val,
                                r2_train_std=r2_train_val_std,
                                rmse_train_std=rmse_train_val_std,
                            )
                            self.add_result(result)
                            # if successful, break out of the retry loop
                            break
                        except Exception as e:
                            self.logger.warning(
                                "Model %s failed on attempt %d/%d: %s",
                                model_cls.__name__,
                                attempt + 1,
                                self.max_retries,
                                str(e),
                            )
                            if attempt == self.max_retries - 1:
                                self.logger.error(
                                    "Model %s failed after %d attempts, skipping.",
                                    model_cls.__name__,
                                    self.max_retries,
                                )

        # Get the best result and log the comparison
        best_result = self.best_result()
        self.log_compare(
            best_model_name=best_result.model_name,
            x_transforms=best_result.x_transforms,
            y_transforms=best_result.y_transforms,
            best_params_for_this_model=best_result.params,
            r2_score=best_result.r2_test,
            rmse_score=best_result.rmse_test,
        )

    def fit_from_reinitialized(
        self,
        x: InputLike,
        y: InputLike,
        result_id: int | None = None,
        random_seed: int | None = None,
        transformed_emulator_params: None | TransformedEmulatorParams = None,
    ) -> TransformedEmulator:
        """
        Fit a fresh model with reinitialized parameters using the best configuration.

        This method creates a new model instance with the same configuration as the
        best (or specified) model from the comparison, but with freshly initialized
        parameters fitted on the provided data.

        Parameters
        ----------
        x: InputLike
            Input features for training the fresh model.
        y: InputLike
            Target values for training the fresh model.
        result_id: int | None
            The ID of the result to use. If None, uses the best model. Defaults to None.
        random_seed: int | None
            Random seed for parameter initialization. Defaults to None.
        transformed_emulator_params: None | TransformedEmulatorParams
            Parameters for the transformed emulator. When None, the same parameters as
            used when identifying the best model are used. Defaults to None.

        Returns
        -------
        TransformedEmulator
            A new model instance with the same configuration but fresh parameters
            fitted on the provided data.

        Notes
        -----
        Unlike TransformedEmulator.refit() which retrains an existing model,
        this method creates a completely new model instance with reinitialized
        parameters. This ensures that when fitting on new data that the same
        initialization conditions are applied. This can have an affect for example
        given kernel initialization in Gaussian Processes or weight initialization in
        neural networks.

        """
        transformed_emulator_params = (
            transformed_emulator_params or self.transformed_emulator_params
        )

        # Get the result to use
        result = self.best_result() if result_id is None else self.get_result(result_id)

        # NOTE: function passes data to the Emulator model which handles conversion to
        # tensors and device handling
        return fit_from_reinitialized(
            x,
            y,
            emulator=result.model,
            transformed_emulator_params=transformed_emulator_params,
            device=str(self.device),
            random_seed=random_seed,
        )

    def plot(  # noqa: PLR0912, PLR0915
        self,
        model_obj: int | Emulator | Result,
        input_index: list[int] | int | None = None,
        output_index: list[int] | int | None = None,
        input_ranges: dict | None = None,
        output_ranges: dict | None = None,
        figsize=None,
        ncols: int = 3,
        fname: str | None = None,
    ):
        """
        Plot the evaluation of the model with the given result_id.

        Parameters
        ----------
        model_obj: int | Emulator | Result
            The model to plot. Can be an integer ID of a Result, an Emulator instance,
            or a Result instance.
        input_index: int
            The index of the input feature to plot against the output.
        output_index: int
            The index of the output feature to plot against the input.
        input_ranges: dict | None
            The ranges of the input features to consider for the plot. Ranges are
            combined such that the final subset is the intersection data within
            the specified ranges. Defaults to None.
        output_ranges: dict | None
            The ranges of the output features to consider for the plot. Ranges are
            combined such that the final subset is the intersection data within
            the specified ranges. Defaults to None.
        figsize: tuple[int, int] | None
            The size of the figure to create. If None, it is set based on the number
            of input and output features.
        ncols: int
            The number of columns in the subplot grid. Defaults to 3.
        fname: str | None
            If provided, the figure will be saved to this file path.
        """
        result = None
        if isinstance(model_obj, int):
            if model_obj not in self._id_to_result:
                raise ValueError(f"No result found with ID: {model_obj}")
            result = self.get_result(model_obj)
            model = result.model
        elif isinstance(model_obj, Emulator):
            model = model_obj
        elif isinstance(model_obj, Result):
            model = model_obj.model

        test_x, test_y = self._convert_to_tensors(self.test)

        # Re-run prediction with just this model to get the predictions
        y_pred, y_variance = model.predict_mean_and_variance(test_x)
        r2_score = evaluate(y_pred, test_y, r2_metric())

        # Handle ranges
        input_ranges = input_ranges or {}
        output_ranges = output_ranges or {}

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
        nrows, ncols = calculate_subplot_layout(n_plots, ncols)

        # Set up the figure
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axs = axs.flatten()

        plot_index = 0
        for out_idx in output_index:
            for in_idx in input_index:
                if plot_index < len(axs):

                    def subset_data_by_ranges(x, y, y_p, ranges, data, data_name):
                        """Subsets data to joint specified ranges on a given array."""
                        for idx, (lower, upper) in ranges.items():
                            if idx < 0 or idx >= data.shape[1]:
                                msg = (
                                    f"Specified {data_name} range index ({idx}) is out "
                                    f"of bounds for {data_name} data of shape "
                                    f"({data.shape})."
                                )
                                raise ValueError(msg)
                            if lower > upper:
                                msg = (
                                    f"Specified {data_name} range ({lower}, {upper}) "
                                    "is invalid. Lower bound must be less than upper "
                                    "bound."
                                )
                                raise ValueError(msg)
                            idxs = (data[:, idx] >= lower) & (data[:, idx] < upper)
                            x, y, y_p, data = x[idxs], y[idxs], y_p[idxs], data[idxs]
                        return x, y, y_p

                    def subset_inputs(x, y, y_p):
                        """Subsets input data to intersection of specified ranges."""
                        return subset_data_by_ranges(
                            x, y, y_p, input_ranges, x, "input"
                        )

                    def subset_outputs(x, y, y_p):
                        """Subsets output data to intersection of specified ranges."""
                        return subset_data_by_ranges(
                            x, y, y_p, output_ranges, y, "output"
                        )

                    # Subset the data to be plotted
                    x, y, y_pred_subset = subset_outputs(
                        *subset_inputs(test_x, test_y, y_pred)
                    )

                    # Generate subplot
                    plot_xy(
                        x[:, in_idx],
                        y[:, out_idx],
                        y_pred_subset[:, out_idx],
                        y_variance[:, out_idx] if y_variance is not None else None,
                        ax=axs[plot_index],
                        title=f"$x_{in_idx}$ vs. $y_{out_idx}$",
                        input_index=in_idx,
                        output_index=out_idx,
                        r2_score=r2_score,
                    )
                    plot_index += 1

        # Hide any unused subplots
        for ax in axs[plot_index:]:
            ax.set_visible(False)
        plt.tight_layout()

        if fname is None:
            return display_figure(fig)
        fig.savefig(fname, bbox_inches="tight")
        return None

    def plot_surface(
        self,
        model: Emulator,
        parameters_range: dict[str, tuple[float, float]],
        input_index_pair: tuple[int, int] | None = None,
        output_index: int | None = None,
        input_ranges: dict[int, tuple[float, float]] | None = None,
        output_range: tuple[float, float] | None = None,
        quantile: float = 0.5,
        figsize=None,
        fname: str | None = None,
    ):
        """Plot the emulator mean and variance over a grid for a pair of parameters.

        This is useful for visualizing the emulator's behavior in 2D slices of the input
        space while keeping other parameters fixed at a specific quantile (default is
        median).

        Parameters
        ----------
        model: Emulator
            The emulator model to plot.
        parameters_range: dict[str, tuple[float, float]]
            A dictionary specifying the ranges for all input parameters. Keys are
            parameter names and values are tuples of (min, max). The dictionary should
            be ordered equivalently to the order of parameters used to train the model.
        input_index_pair: tuple[int, int] | None
            A tuple of two integers specifying the indices of the input parameters to
            plot. If None, the first two parameters (0, 1) are used. Defaults to None.
        output_index: int | None
            The index of the output to plot. If None, the first output (0) is used.
            Defaults to None.
        input_ranges: dict[int, tuple[float, float]] | None
            A dictionary specifying the ranges for input parameters to consider.
            Keys are parameter indices and values are tuples of (min, max). If None,
            the full range from the simulator is used. Defaults to None.
        output_range: tuple[float, float] | None
            A tuple specifying the (min, max) range for the output to consider. If None,
            the full range from the simulator is used. Defaults to None.
        quantile: float
            The quantile of the other input parameters to fix when plotting the 2D
            slice. Must be between 0 and 1. Defaults to 0.5.
        figsize: tuple[int, int] | None
            The size of the figure to create. If None, a default size is used.
            Defaults to None.
        fname: str | None
            If provided, the figure will be saved to this file path. If None, the figure
            will be displayed. Defaults to None.
        """
        # Update parameter ranges if provided
        if input_ranges is not None:
            # Copy to avoid modifying the parameters_range passed
            parameters_range = parameters_range.copy()
            parameters_range.update(
                {
                    list(parameters_range.keys())[k]: input_ranges[k]
                    for k in input_ranges
                }
            )

        fig, _ = create_and_plot_slice(
            model,
            parameters_range,
            input_index_pair if input_index_pair is not None else (0, 1),
            output_idx=output_index if output_index is not None else 0,
            vmin=None if output_range is None else output_range[0],
            vmax=None if output_range is None else output_range[1],
            quantile=quantile,
        )
        if figsize is not None:
            fig.set_size_inches(figsize)
        if fname is None:
            return display_figure(fig)
        fig.savefig(fname, bbox_inches="tight")
        return None

    def save(
        self,
        model_obj: int | Emulator | Result,
        path: str | Path | None = None,
        use_timestamp: bool = True,
    ) -> Path:
        """Save model to disk.

        Parameters
        ----------
        model_obj : int | Emulator | Result
            The model to save. Can be an integer ID of a Result, an Emulator instance,
            or a Result instance.
        path : str
            Path to save the model.
        use_timestamp : bool
            If True, appends a timestamp to the filename to ensure uniqueness.
        """
        result = None
        if isinstance(model_obj, int):
            if model_obj not in self._id_to_result:
                raise ValueError(f"No result found with ID: {model_obj}")
            result = self.get_result(model_obj)
            model = result.model
            model_name = result.model_name
        elif isinstance(model_obj, Emulator):
            model = model_obj
            if isinstance(model_obj, TransformedEmulator):
                model_name = model_obj.untransformed_model_name
            else:
                model_name = model.model_name()
        elif isinstance(model_obj, Result):
            model = model_obj.model
            model_name = model_obj.model_name
            result = model_obj

        # Create a unique filename based on the model name, id and date
        filename = f"{model_name}_{result.id}" if result is not None else model_name
        if use_timestamp:
            t = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename += f"_{t}"

        if result is not None:
            return self.model_serialiser._save_result(result, filename, path)
        return self.model_serialiser._save_model(model, filename, path)

    def load(self, path: str | Path) -> Emulator | Result:
        """Load a stored model or result from disk.

        Parameters
        ----------
        path : str
            Path to model.

        Returns
        -------
        Emulator | Result
            The loaded model or result object.
        """
        return self.model_serialiser._load_result(path)

    @staticmethod
    def load_model(path: str | Path) -> Emulator:
        """Load a stored model directly from a given path.

        Parameters
        ----------
        path : str | Path
            Path to the model.

        Returns
        -------
        Emulator
            The loaded model object.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        if isinstance(path, Path):
            path = str(path)
        if not path.endswith(".joblib"):
            path += ".joblib"
        return joblib.load(path)
