import math

import torch
from torch import Tensor

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
        """
        self.emulator = emulator
        self.supports_grad = emulator.supports_grad
        if not 0 < alpha < 1:
            msg = "Conformal coverage level alpha must be in (0, 1)."
            raise ValueError(msg)
        self.alpha = alpha  # desired predictive coverage (e.g., 0.95)
        self.n_samples = 1000  # number of samples to draw from base emulator if needed
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
        self.emulator.fit(x, y, validation_data=None)

        with torch.no_grad():
            if validation_data is None:
                msg = "Conformal emulator requires calibration data for quantiles."
                raise ValueError(msg)

            # Destructure calibration data
            x_cal, y_true_cal = validation_data

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
    """
    Conformal UQ with an MLP.

    This class is to provide ensemble of MLP emulators, each initialized with the same
    input and output data.

    """

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        alpha: float = 0.95,
        standardize_x: bool = True,
        standardize_y: bool = True,
        device: DeviceLike | None = None,
        **mlp_kwargs,
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
        mlp_kwargs: dict | None
            Additional keyword arguments for the MLP constructor.
        """
        PyTorchBackend.__init__(self)
        self.mlp_kwargs = mlp_kwargs or {}
        emulator = MLP(
            x,
            y,
            standardize_x=standardize_x,
            standardize_y=standardize_y,
            device=device,
            **self.mlp_kwargs,
        )
        Conformal.__init__(self, emulator=emulator, alpha=alpha, device=device)

    @staticmethod
    def is_multioutput() -> bool:
        """Ensemble of MLPs supports multi-output."""
        return True

    @staticmethod
    def get_tune_params() -> TuneParams:
        """Return a dictionary of hyperparameters to tune."""
        return MLP.get_tune_params()
