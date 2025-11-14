import math
import sys
from collections.abc import Callable
from typing import Literal

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import DeviceLike, DistributionLike, TensorLike, TuneParams
from autoemulate.emulators.base import Emulator, PyTorchBackend
from autoemulate.emulators.nn.mlp import MLP, _generate_mlp_docstring


class QuantileLoss(nn.Module):
    """Quantile loss for quantile regression.

    This loss function asymmetrically penalizes over- and under-predictions, enabling
    the model to learn specific quantiles of the conditional distribution.
    """

    def __init__(self, quantile: float):
        """Initialize quantile loss.

        Parameters
        ----------
        quantile: float
            Target quantile level in (0, 1). For example, 0.1 for 10th percentile, 0.5
            for median, 0.9 for 90th percentile.
        """
        super().__init__()
        if not 0 < quantile < 1:
            msg = f"Quantile must be in (0, 1), got {quantile}"
            raise ValueError(msg)
        self.quantile = quantile

    def forward(self, y_pred: TensorLike, y_true: TensorLike) -> TensorLike:
        """Compute quantile loss.

        Parameters
        ----------
        y_pred: TensorLike
            Predicted values.
        y_true: TensorLike
            True target values.

        Returns
        -------
        TensorLike
            Scalar loss value.
        """
        errors = y_true - y_pred
        # Mean across batch and targets
        return torch.max(self.quantile * errors, (self.quantile - 1) * errors).mean()


class QuantileMLP(MLP):
    """MLP with quantile loss for quantile regression."""

    def __init__(self, quantile: float, **kwargs):
        """Initialize quantile MLP.

        Parameters
        ----------
        quantile: float
            Target quantile level in (0, 1).
        **kwargs
            Keyword arguments passed to MLP parent class.
        """
        super().__init__(**kwargs)
        self.quantile = quantile
        self.quantile_loss = QuantileLoss(quantile)

    def loss_func(self, y_pred, y_true):
        """Quantile loss function."""
        return self.quantile_loss(y_pred, y_true)


class Conformal(Emulator):
    """Conformal Uncertainty Quantification (UQ) wrapper for emulators.

    This class wraps a base emulator to provide conformal prediction intervals with
    guaranteed frequentist coverage.

    Both standard split conformal and Conformalized Quantile Regression (CQR) methods
    are supported.

    Conformalized Quantile Regression (CQR) is defaultly implemented with two neural net
    quantile regressors predicting lower and upper quantiles, followed by a calibration
    step to ensure valid coverage. Note the _fit_quantile_regressors method can be
    overridden to implement custom quantile regressors.

    Additional methods for input-dependent intervals (such as scaling) can be
    implemented by adding further supported "method" strings and providing corresponding
    logic in the _fit and _predict methods.

    References
    ----------
    - Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized Quantile Regression.
      In Advances in Neural Information Processing Systems (Vol. 32).
      https://papers.nips.cc/paper/8613-conformalized-quantile-regression.pdf

    """

    supports_uq = True

    def __init__(
        self,
        emulator: Emulator,
        alpha: float = 0.95,
        device: DeviceLike | None = None,
        calibration_ratio: float = 0.2,
        n_samples: int = 1000,
        method: Literal["constant", "quantile"] = "constant",
        to_distribution: Callable[
            [TensorLike | None, tuple[TensorLike, TensorLike]], DistributionLike
        ] = lambda _mean, bounds: torch.distributions.Uniform(bounds[0], bounds[1]),
        quantile_emulator_kwargs: dict | None = None,
    ):
        """Initialize a conformal emulator.

        Parameters
        ----------
        emulator: Emulator
            Base emulator to wrap for conformal UQ.
        alpha: float
            Desired predictive coverage level (e.g., 0.95 for 95% coverage). Must be in
            (0, 1).
        device: DeviceLike | None
            Device to run the model on (e.g., "cpu", "cuda"). Defaults to None.
        calibration_ratio: float
            Fraction of the training data to reserve for calibration if explicit
            validation data is not provided. Must lie in (0, 1). Defaults to 0.2.
        n_samples: int
            Number of samples used for sampling-based predictions or internal
            procedures. Defaults to 1000.
        method: Literal["constant", "quantile"]
            Conformalization method to use:
            - "constant": Standard split conformal with constant-width intervals
            - "quantile": Conformalized Quantile Regression (CQR) with input-dependent
              intervals. Defaults to "constant".
        to_distribution: Callable[[TensorLike | None, tuple[TensorLike, TensorLike]], DistributionLike]
            A callable that takes an optional mean and a tuple of lower and upper bounds
            as input and returns a distribution over that interval.
            Defaults to lambda _mean, bounds: torch.distributions.Uniform(bounds[0], bounds[1]).
        quantile_emulator_kwargs: dict | None
            Additional keyword arguments for the quantile emulators when
            method="quantile". Defaults to None.
        """  # noqa: E501
        self.emulator = emulator
        self.supports_grad = emulator.supports_grad
        if not 0 < alpha < 1:
            msg = "Conformal coverage level alpha must be in (0, 1)."
            raise ValueError(msg)
        if not 0 < calibration_ratio < 1:
            msg = "Calibration ratio must lie strictly between 0 and 1."
            raise ValueError(msg)
        if method not in {"constant", "quantile"}:
            msg = f"Method must be 'constant' or 'quantile', got '{method}'."
            raise ValueError(msg)
        self.alpha = alpha  # desired predictive coverage (e.g., 0.95)
        self.calibration_ratio = calibration_ratio
        self.n_samples = n_samples
        self.method = method
        self.to_distribution = to_distribution
        self.quantile_emulator_kwargs = quantile_emulator_kwargs or {}
        TorchDeviceMixin.__init__(self, device=device)
        self.supports_grad = emulator.supports_grad

    @staticmethod
    def is_multioutput() -> bool:
        """Ensemble supports multi-output."""
        return True

    @staticmethod
    def get_tune_params() -> TuneParams:
        """Return a dictionary of hyperparameters to tune."""
        return {}

    def _fit(
        self,
        x: TensorLike,
        y: TensorLike,
        validation_data: tuple[TensorLike, TensorLike] | None = None,
    ):
        x_train, y_train = x, y

        # If not validation data passed, take random permutation of training data and
        # hold out a calibration set according to calibration_ratio
        if validation_data is None:
            n_samples = x.shape[0]
            if n_samples < 2:
                msg = "At least two samples are required to create a calibration split."
                raise ValueError(msg)

            n_cal = max(1, math.ceil(n_samples * self.calibration_ratio))
            if n_cal >= n_samples:
                n_cal = n_samples - 1
            perm = torch.randperm(n_samples, device=x.device)
            cal_idx = perm[:n_cal]
            train_idx = perm[n_cal:]
            if train_idx.numel() == 0:
                msg = "Calibration split left no samples for training."
                raise ValueError(msg)
            x_cal = x[cal_idx]
            y_true_cal = y[cal_idx]
            x_train = x[train_idx]
            y_train = y[train_idx]
        else:
            x_cal, y_true_cal = validation_data

        # Fit the base emulator
        self.emulator.fit(x_train, y_train, validation_data=None)

        n_cal = x_cal.shape[0]
        # Check calibration data is non-empty
        if n_cal == 0:
            msg = "Calibration set must contain at least one sample."
            raise ValueError(msg)

        with torch.no_grad():
            # Predict and calculate residuals
            y_pred_cal = self.output_to_tensor(self.emulator.predict(x_cal))

        # Standard split conformal for constant-width intervals
        if self.method == "constant":
            # Compute absolute residuals
            residuals = torch.abs(y_true_cal - y_pred_cal)

            # Apply finite-sample correction to quantile level
            quantile_level = min(1.0, math.ceil((n_cal + 1) * self.alpha) / n_cal)

            # Calibrate over the batch dim with a separate quantile per output
            self.q = torch.quantile(residuals, quantile_level, dim=0)

        # Conformalized Quantile Regression for input-dependent intervals
        elif self.method == "quantile":
            # Train quantile regressors
            self._fit_quantile_regressors(x_train, y_train, x_cal, y_true_cal)

        self.is_fitted_ = True

    def _fit_quantile_regressors(
        self,
        x_train: TensorLike,
        y_train: TensorLike,
        x_cal: TensorLike,
        y_true_cal: TensorLike,
    ):
        """Fit quantile regressors for CQR method.

        Trains two quantile regressors to predict lower and upper quantiles,
        then calibrates the width using the calibration set.
        """
        # Calculate quantile levels
        lower_q = (1 - self.alpha) / 2
        upper_q = 1 - lower_q

        # Lower quantile emulator
        self.lower_quantile_emulator = QuantileMLP(
            lower_q,
            x=x_train,
            y=y_train,
            device=self.device,
            **self.quantile_emulator_kwargs,
        )

        # Upper quantile emulator
        self.upper_quantile_emulator = QuantileMLP(
            upper_q,
            x=x_train,
            y=y_train,
            device=self.device,
            **self.quantile_emulator_kwargs,
        )

        # Fit the quantile emulators
        self.lower_quantile_emulator.fit(x_train, y_train, validation_data=None)
        self.upper_quantile_emulator.fit(x_train, y_train, validation_data=None)

        # Predict quantiles on calibration set
        with torch.no_grad():
            lower_pred_cal = self.output_to_tensor(
                self.lower_quantile_emulator.predict(x_cal)
            )
            upper_pred_cal = self.output_to_tensor(
                self.upper_quantile_emulator.predict(x_cal)
            )

            # Calculate conformalization scores (non-conformity scores)
            # For CQR, the score is max(lower - y, y - upper)
            scores = torch.maximum(
                lower_pred_cal - y_true_cal, y_true_cal - upper_pred_cal
            )

            # Apply finite-sample correction
            n_cal = x_cal.shape[0]
            quantile_level = min(1.0, math.ceil((n_cal + 1) * self.alpha) / n_cal)

            # Compute the correction term per output dimension
            self.q_cqr = torch.quantile(scores, quantile_level, dim=0)

    def _predict(self, x: TensorLike, with_grad: bool) -> DistributionLike:
        # Standard split conformal: constant width intervals
        if self.method == "constant":
            pred = self.emulator.predict(x, with_grad)
            mean = self.output_to_tensor(pred)
            q = self.q.to(mean.device)
            return torch.distributions.Independent(
                self.to_distribution(None, (mean - q, mean + q)),
                reinterpreted_batch_ndims=mean.ndim - 1,
            )

        # Conformalized Quantile Regression: input-dependent intervals
        if self.method == "quantile":
            lower_pred = self.output_to_tensor(
                self.lower_quantile_emulator.predict(x, with_grad)
            )
            upper_pred = self.output_to_tensor(
                self.upper_quantile_emulator.predict(x, with_grad)
            )
            q_cqr = self.q_cqr.to(lower_pred.device)

            # Apply calibration correction
            lower_bound = lower_pred - q_cqr
            upper_bound = upper_pred + q_cqr

            # Return uniform distribution over the calibrated interval
            return torch.distributions.Independent(
                self.to_distribution(None, (lower_bound, upper_bound)),
                reinterpreted_batch_ndims=lower_bound.ndim - 1,
            )

        msg = f"Unknown method: {self.method}"
        raise ValueError(msg)


class ConformalMLP(Conformal, PyTorchBackend):
    """Conformal UQ with an MLP.

    This class is to provides UQ via conformal prediction intervals wrapped around a
    Multi-Layer Perceptron (MLP) emulator.

    Both standard split conformal and Conformalized Quantile Regression (CQR) methods
    are supported.

    """

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        standardize_x: bool = True,
        standardize_y: bool = True,
        activation_cls: type[nn.Module] = nn.ReLU,
        loss_fn_cls: type[nn.Module] = nn.MSELoss,
        epochs: int = 100,
        batch_size: int = 16,
        layer_dims: list[int] | None = None,
        weight_init: str = "default",
        scale: float = 1.0,
        bias_init: str = "default",
        dropout_prob: float | None = None,
        lr: float = 1e-2,
        params_size: int = 1,
        random_seed: int | None = None,
        device: DeviceLike | None = None,
        scheduler_cls: type[LRScheduler] | None = None,
        scheduler_params: dict | None = None,
        alpha: float = 0.95,
        calibration_ratio: float = 0.2,
        method: Literal["constant", "quantile"] = "constant",
        quantile_emulator_kwargs: dict | None = None,
    ):
        nn.Module.__init__(self)

        # Construct docstring
        conformal_kwargs = """
        alpha: float
            Desired predictive coverage level forwarded to the conformal wrapper.
        calibration_ratio: float
            Fraction of training samples to hold out for calibration when an explicit
            validation set is not provided.
        method: Literal["constant", "quantile"]
            Conformalization method:
            - "constant": Standard split conformal (constant-width intervals)
            - "quantile": Conformalized Quantile Regression (input-dependent intervals)
            Defaults to "constant".
        quantile_emulator_kwargs: dict | None
            Additional keyword arguments for the quantile emulators when
            method="quantile". Defaults to None.
        """
        conformal_mlp_params = _generate_mlp_docstring(
            additional_parameters_docstring=conformal_kwargs,
            default_dropout_prob=None,
        )
        self.__doc__ = (
            """    Initialize a conformal MLP emulator.\n\n""" + conformal_mlp_params
        )

        emulator = MLP(
            x,
            y,
            standardize_x=standardize_x,
            standardize_y=standardize_y,
            device=device,
            activation_cls=activation_cls,
            loss_fn_cls=loss_fn_cls,
            epochs=epochs,
            batch_size=batch_size,
            layer_dims=layer_dims,
            weight_init=weight_init,
            scale=scale,
            bias_init=bias_init,
            dropout_prob=dropout_prob,
            lr=lr,
            params_size=params_size,
            random_seed=random_seed,
            scheduler_cls=scheduler_cls,
            scheduler_params=scheduler_params,
        )

        quantile_defaults = {
            "standardize_x": standardize_x,
            "standardize_y": standardize_y,
            "activation_cls": activation_cls,
            "loss_fn_cls": loss_fn_cls,
            "epochs": epochs,
            "batch_size": batch_size,
            "layer_dims": layer_dims,
            "weight_init": weight_init,
            "scale": scale,
            "bias_init": bias_init,
            "dropout_prob": dropout_prob,
            "lr": lr,
            "params_size": params_size,
            "random_seed": random_seed,
            "scheduler_cls": scheduler_cls,
            "scheduler_params": scheduler_params,
        }
        merged_quantile_kwargs = {
            **quantile_defaults,
            **(quantile_emulator_kwargs or {}),
        }
        Conformal.__init__(
            self,
            emulator=emulator,
            alpha=alpha,
            device=device,
            calibration_ratio=calibration_ratio,
            method=method,
            quantile_emulator_kwargs=merged_quantile_kwargs,
        )

    @staticmethod
    def is_multioutput() -> bool:
        """Ensemble of MLPs supports multi-output."""
        return True

    @staticmethod
    def get_tune_params() -> TuneParams:
        """Return a dictionary of hyperparameters to tune."""
        return MLP.get_tune_params()


def create_conformal_subclass(
    name: str,
    conformal_mlp_base_class: type[ConformalMLP],
    method: Literal["constant", "quantile"],
    auto_register: bool = True,
    overwrite: bool = True,
    **fixed_kwargs,
) -> type[ConformalMLP]:
    """
    Create a subclass of ConformalMLP with given fixed_kwargs.

    This function creates a subclass of ConformalMLP where certain parameters
    are fixed to specific values, reducing the parameter space for tuning.

    The created subclass is automatically registered with the main emulator Registry
    (unless auto_register=False), making it discoverable by AutoEmulate.

    Parameters
    ----------
    name: str
        Name for the created subclass.
    conformal_mlp_base_class: type[ConformalMLP]
        Base class to inherit from (typically ConformalMLP).
    method: Literal["constant", "quantile"]
        Conformalization method to use in the subclass.
    auto_register : bool
        Whether to automatically register the created subclass with the main emulator
        Registry. Defaults to True.
    overwrite : bool
        Whether to allow overwriting an existing class with the same name in the
        main Registry. Useful for interactive development in notebooks. Defaults to
        True.
    **fixed_kwargs
        Keyword arguments to fix in the subclass. These parameters will be
        set to the provided values and excluded from hyperparameter tuning.

    Returns
    -------
    type[ConformalMLP]
        A new subclass of ConformalMLP with the specified parameters fixed.
        The returned class can be pickled and used like any other GP emulator.

    Raises
    ------
    ValueError
        If `name` matches `model_name()` or `short_name()` of an already registered
        emulator in the main Registry and `overwrite=False`.

    Notes
    -----
    - Fixed parameters are automatically excluded from `get_tune_params()` to prevent
    them from being included in hyperparameter optimization.
    - Pickling: The created subclass is registered in the caller's module namespace,
    ensuring it can be pickled and unpickled correctly even when created in downstream
    code that uses autoemulate as a dependency.
    - If auto_register=True (default), the class is also added to the main Registry.
    """
    standardize_x = fixed_kwargs.get("standardize_x", True)
    standardize_y = fixed_kwargs.get("standardize_y", True)
    activation_cls: type[nn.Module] = fixed_kwargs.get("activation_cls", nn.ReLU)
    loss_fn_cls: type[nn.Module] = fixed_kwargs.get("loss_fn_cls", nn.MSELoss)
    epochs: int = fixed_kwargs.get("epochs", 100)
    batch_size: int = fixed_kwargs.get("batch_size", 16)
    layer_dims: list[int] | None = fixed_kwargs.get("layer_dims")
    weight_init: str = fixed_kwargs.get("weight_init", "default")
    scale: float = fixed_kwargs.get("scale", 1.0)
    bias_init: str = fixed_kwargs.get("bias_init", "default")
    dropout_prob: float | None = fixed_kwargs.get("dropout_prob")
    lr: float = fixed_kwargs.get("lr", 1e-2)
    params_size: int = fixed_kwargs.get("params_size", 1)
    random_seed: int | None = fixed_kwargs.get("random_seed")
    device: DeviceLike | None = fixed_kwargs.get("device")
    scheduler_cls: type[LRScheduler] | None = fixed_kwargs.get("scheduler_cls")
    scheduler_params: dict | None = fixed_kwargs.get("scheduler_params")
    alpha: float = fixed_kwargs.get("alpha", 0.95)
    calibration_ratio: float = fixed_kwargs.get("calibration_ratio", 0.2)
    quantile_emulator_kwargs: dict | None = fixed_kwargs.get("quantile_emulator_kwargs")

    class ConformalMLPSubclass(conformal_mlp_base_class):
        def __init__(
            self,
            x: TensorLike,
            y: TensorLike,
            standardize_x: bool = standardize_x,
            standardize_y: bool = standardize_y,
            activation_cls: type[nn.Module] = activation_cls,
            loss_fn_cls: type[nn.Module] = loss_fn_cls,
            epochs: int = epochs,
            batch_size: int = batch_size,
            layer_dims: list[int] | None = layer_dims,
            weight_init: str = weight_init,
            scale: float = scale,
            bias_init: str = bias_init,
            dropout_prob: float | None = dropout_prob,
            lr: float = lr,
            params_size: int = params_size,
            random_seed: int | None = random_seed,
            device: DeviceLike | None = device,
            scheduler_cls: type[LRScheduler] | None = scheduler_cls,
            scheduler_params: dict | None = scheduler_params,
            alpha: float = alpha,
            calibration_ratio: float = calibration_ratio,
            method: Literal["constant", "quantile"] = method,
            quantile_emulator_kwargs: dict | None = quantile_emulator_kwargs,
        ):
            super().__init__(
                x,
                y,
                standardize_x=standardize_x,
                standardize_y=standardize_y,
                activation_cls=activation_cls,
                loss_fn_cls=loss_fn_cls,
                epochs=epochs,
                batch_size=batch_size,
                layer_dims=layer_dims,
                weight_init=weight_init,
                scale=scale,
                bias_init=bias_init,
                dropout_prob=dropout_prob,
                lr=lr,
                params_size=params_size,
                random_seed=random_seed,
                device=device,
                scheduler_cls=scheduler_cls,
                scheduler_params=scheduler_params,
                alpha=alpha,
                calibration_ratio=calibration_ratio,
                method=method,
                quantile_emulator_kwargs=quantile_emulator_kwargs,
            )

        @staticmethod
        def get_tune_params():
            """Get tunable parameters, excluding those that are fixed."""
            tune_params = conformal_mlp_base_class.get_tune_params()
            # Remove fixed parameters from tuning
            tune_params.pop("method", None)
            for key in fixed_kwargs:
                tune_params.pop(key, None)
            return tune_params

    # Create a more descriptive docstring that includes fixed parameters
    method_and_fixed_kwargs = {
        **fixed_kwargs,
    }
    fixed_params_str = "\n    ".join(
        f"- {k} = {v.__name__ if callable(v) else v}"
        for k, v in method_and_fixed_kwargs.items()
    )

    ConformalMLPSubclass.__doc__ = f"""
    {conformal_mlp_base_class.__doc__}

    Notes
    -----
    {name} is a subclass of {conformal_mlp_base_class.__name__} and has the following
    parameters set during initialization:
    {fixed_params_str}

    For any parameters set with this approach, they are also excluded from the search
    space when tuning. For example, if the `method` is set to `constant`,
    the "constant" method will always be used as the `method`. Note that in this case
    the associated hyperparameters (such as lengthscale) will still be fitted during
    model training and are not fixed.
    """

    # Determine the caller's module for proper pickling support.
    # When called from autoemulate itself, use __name__.
    # When called from user code, use the caller's module
    caller_frame = sys._getframe(1)
    caller_module_name = caller_frame.f_globals.get("__name__", __name__)

    # Set the class name and module
    ConformalMLPSubclass.__name__ = name
    ConformalMLPSubclass.__qualname__ = name
    ConformalMLPSubclass.__module__ = caller_module_name

    # Register class in the caller's module globals for pickling
    # This ensures the class can be pickled/unpickled correctly
    caller_frame.f_globals[name] = ConformalMLPSubclass
    # Also register in the caller's module if it's a real module (not __main__)
    if caller_module_name in sys.modules and caller_module_name != "__main__":
        setattr(sys.modules[caller_module_name], name, ConformalMLPSubclass)

    # Automatically register with the main emulator Registry if requested
    if auto_register:
        # Lazy import to avoid circular dependency with __init__.py
        from autoemulate.emulators import register  # noqa: PLC0415

        register(ConformalMLPSubclass, overwrite=overwrite)

    return ConformalMLPSubclass


# Built-in GP subclasses - auto_register=False as already registered in Registry init:
# autoemulate/emulators/__init__.py
ConformalMLPConstant = create_conformal_subclass(
    "ConformalMLPConstant",
    ConformalMLP,
    method="constant",
    auto_register=False,
)
ConformalMLPQuantile = create_conformal_subclass(
    "ConformalMLPQuantile",
    ConformalMLP,
    method="quantile",
    auto_register=False,
)
