import math

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import DeviceLike, DistributionLike, TensorLike, TuneParams
from autoemulate.emulators.base import Emulator, PyTorchBackend
from autoemulate.emulators.nn.mlp import MLP


class Conformal(Emulator):
    """Conformal Uncertainty Quantification (UQ) wrapper for emulators.

    This class wraps a base emulator to provide conformal prediction intervals with
    guaranteed frequentist coverage.
    """

    supports_uq = True

    def __init__(
        self,
        emulator: Emulator,
        alpha: float = 0.95,
        device: DeviceLike | None = None,
        calibration_ratio: float = 0.2,
        n_samples: int = 1000,
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
            Number of samples used for sampling-based predictions or
            internal procedures. Defaults to 1000.
        """
        self.emulator = emulator
        self.supports_grad = emulator.supports_grad
        if not 0 < alpha < 1:
            msg = "Conformal coverage level alpha must be in (0, 1)."
            raise ValueError(msg)
        if not 0 < calibration_ratio < 1:
            msg = "Calibration ratio must lie strictly between 0 and 1."
            raise ValueError(msg)
        self.alpha = alpha  # desired predictive coverage (e.g., 0.95)
        self.calibration_ratio = calibration_ratio
        self.n_samples = n_samples
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
    ) -> None:
        x_train, y_train = x, y
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

        self.emulator.fit(x_train, y_train, validation_data=None)

        with torch.no_grad():
            n_cal = x_cal.shape[0]
            # Check calibration data is non-empty
            if n_cal == 0:
                msg = "Calibration set must contain at least one sample."
                raise ValueError(msg)

            # Predict and calculate residuals
            y_pred_cal = self.output_to_tensor(self.emulator.predict(x_cal))
            residuals = torch.abs(y_true_cal - y_pred_cal)

            # Apply finite-sample correction to quantile level to ensure valid coverage
            quantile_level = min(1.0, math.ceil((n_cal + 1) * self.alpha) / n_cal)

            # Calibrate over the batch dim with a separate quantile for each output
            self.q = torch.quantile(residuals, quantile_level, dim=0)

        self.is_fitted_ = True

    def _predict(self, x: Tensor, with_grad: bool) -> DistributionLike:
        pred = self.emulator.predict(x, with_grad)
        mean = self.output_to_tensor(pred)
        q = self.q.to(mean.device)
        return torch.distributions.Independent(
            torch.distributions.Uniform(mean - q, mean + q),
            reinterpreted_batch_ndims=mean.ndim - 1,
        )


class ConformalMLP(Conformal, PyTorchBackend):
    """Conformal UQ with an MLP.

    This class is to provide ensemble of MLP emulators, each initialized with the same
    input and output data.

    """

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        standardize_x: bool = True,
        standardize_y: bool = True,
        device: DeviceLike | None = None,
        alpha: float = 0.95,
        calibration_ratio: float = 0.2,
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
        scheduler_cls: type[LRScheduler] | None = None,
        scheduler_params: dict | None = None,
    ):
        """
        Initialize an ensemble of MLPs.

        Parameters
        ----------
        x: TensorLike
            Input data tensor of shape (batch_size, n_features).
        y: TensorLike
            Target values tensor of shape (batch_size, n_outputs).
        standardize_x: bool
            Whether to standardize the input data. Defaults to True.
        standardize_y: bool
            Whether to standardize the output data. Defaults to True.
        device: DeviceLike | None
            Device to run the model on (e.g., "cpu", "cuda"). Defaults to None.
        alpha: float
            Desired predictive coverage level forwarded to the conformal wrapper.
        calibration_ratio: float
            Fraction of training samples to hold out for calibration when an explicit
            validation set is not provided.
        activation_cls: type[nn.Module]
            Activation function to use in the hidden layers. Defaults to `nn.ReLU`.
        loss_fn_cls: type[nn.Module]
            Loss function class used to construct the loss function for training.
            Defaults to `nn.MSELoss`.
        layer_dims: list[int] | None
            Dimensions of the hidden layers. If None, defaults to [32, 16].
            Defaults to None.
        weight_init: str
            Weight initialization method. Options are "default", "normal", "uniform",
            "zeros", "ones", "xavier_uniform", "xavier_normal", "kaiming_uniform",
            "kaiming_normal". Defaults to "default".
        scale: float
            Scale parameter for weight initialization methods. Used as:
            - gain for Xavier methods
            - std for normal distribution
            - bound for uniform distribution (range: [-scale, scale])
            - ignored for Kaiming methods (uses optimal scaling)
            Defaults to 1.0.
        bias_init: str
            Bias initialization method. Options: "zeros", "default":
                - "zeros" initializes biases to zero
                - "default" uses PyTorch's default uniform initialization
        dropout_prob: float | None
            Dropout probability for regularization. If None, no dropout is applied.
            Defaults to None.
        lr: float
            Learning rate for the optimizer. Defaults to 1e-2.
        params_size: int
            Number of parameters to predict per output dimension. Defaults to 1.
        random_seed: int | None
            Random seed for reproducibility. If None, no seed is set. Defaults to None.
        scheduler_cls: type[LRScheduler] | None
            Learning rate scheduler class. If None, no scheduler is used. Defaults to
            None.
        scheduler_params: dict | None
            Additional keyword arguments related to the scheduler.
        """
        nn.Module.__init__(self)

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
        Conformal.__init__(
            self,
            emulator=emulator,
            alpha=alpha,
            device=device,
            calibration_ratio=calibration_ratio,
        )

    @staticmethod
    def is_multioutput() -> bool:
        """Ensemble of MLPs supports multi-output."""
        return True

    @staticmethod
    def get_tune_params() -> TuneParams:
        """Return a dictionary of hyperparameters to tune."""
        return MLP.get_tune_params()
