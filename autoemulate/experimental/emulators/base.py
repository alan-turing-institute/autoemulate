import random
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler

from autoemulate.experimental.data.preprocessors import Preprocessor
from autoemulate.experimental.data.utils import (
    ConversionMixin,
    ValidationMixin,
)
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.types import (
    DistributionLike,
    GaussianLike,
    GaussianProcessLike,
    NumpyLike,
    OutputLike,
    TensorLike,
    TuneConfig,
)


class Emulator(ABC, ValidationMixin, ConversionMixin, TorchDeviceMixin):
    """
    The interface containing methods on emulators that are
    expected by downstream dependents. This includes:
    - `AutoEmulate`
    """

    is_fitted_: bool = False

    @abstractmethod
    def _fit(self, x: TensorLike, y: TensorLike): ...

    def fit(self, x: TensorLike, y: TensorLike):
        self._check(x, y)
        x, y = self._move_tensors_to_device(x, y)
        self._fit(x, y)
        self.is_fitted_ = True

    @abstractmethod
    def __init__(
        self, x: TensorLike | None = None, y: TensorLike | None = None, **kwargs
    ): ...

    @classmethod
    def model_name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def _predict(self, x: TensorLike) -> OutputLike:
        pass

    def predict(self, x: TensorLike) -> OutputLike:
        if not self.is_fitted_:
            msg = "Model is not fitted yet. Call fit() before predict()."
            raise RuntimeError(msg)
        self._check(x, None)
        (x,) = self._move_tensors_to_device(x)
        output = self._predict(x)
        self._check_output(output)
        return output

    @staticmethod
    @abstractmethod
    def is_multioutput() -> bool:
        """Flag to indicate if the model is multioutput or not."""

    @staticmethod
    def get_tune_config() -> TuneConfig:
        """
        The keys in the TuneConfig must be implemented as keyword arguments in the
        __init__ method of any subclasses.

        e.g.

        tune_config: TuneConfig = {
            "lr": list[0.01, 0.1],
            "batch_size": [16, 32],
            "mean"
        }

        model_config: ModelConfig = {
            "lr": 0.01,
            "batch_size": 16
        }

        class MySubClass(Emulator):
            def __init__(lr, batch_size):
                self.lr = lr
                self.batch_size = batch_size
        """
        msg = (
            "Subclasses should implement for generating tuning config specific to "
            "each subclass."
        )
        raise NotImplementedError(msg)

    @classmethod
    def get_random_config(cls):
        return {
            k: v[np.random.randint(len(v))] for k, v in cls.get_tune_config().items()
        }


class DeterministicEmulator(Emulator):
    """An emulator subclass that predicts with deterministic outputs returning a
    `TensorLike`.
    """

    @abstractmethod
    def _predict(self, x: TensorLike) -> TensorLike: ...
    def predict(self, x: TensorLike) -> TensorLike:
        pred = super().predict(x)
        assert isinstance(pred, TensorLike)
        return pred


class ProbabilisticEmulator(Emulator):
    """An emulator subclass that predicts with probabilistic outputs returning a
    `DistributionLike`.
    """

    @abstractmethod
    def _predict(self, x: TensorLike) -> DistributionLike: ...
    def predict(self, x: TensorLike) -> DistributionLike:
        pred = super().predict(x)
        assert isinstance(pred, DistributionLike)
        return pred

    def predict_mean_and_variance(self, x: TensorLike) -> tuple[TensorLike, TensorLike]:
        """Predict mean and variance from the probabilistic output.

        Parameters
        ----------
        x : TensorLike
            Input tensor to make predictions for.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            The emulator predicted mean and variance for `x`.

        """
        pred = self.predict(x)
        return pred.mean, pred.variance


class GaussianEmulator(ProbabilisticEmulator):
    """An emulator subclass that predicts with Gaussian outputs returning a
    `GaussianLike`.
    """

    @abstractmethod
    def _predict(self, x: TensorLike) -> GaussianLike: ...
    def predict(self, x: TensorLike) -> GaussianLike:
        pred = super().predict(x)
        assert isinstance(pred, GaussianLike)
        return pred


class GaussianProcessEmulator(GaussianEmulator):
    """A Gaussian Process emulator subclass that predicts with output
    `GaussianProcessLike`.
    """

    @abstractmethod
    def _predict(self, x: TensorLike) -> GaussianProcessLike: ...
    def predict(self, x: TensorLike) -> GaussianProcessLike:
        pred = super().predict(x)
        assert isinstance(pred, GaussianProcessLike)
        return pred


class PyTorchBackend(nn.Module, Emulator, Preprocessor):
    """
    PyTorchBackend is a torch model and implements the base class.
    This provides default implementations to further subclasses.
    This means that models can subclass and only need to implement
    `.forward()` to have an emulator to be run in `AutoEmulate`
    """

    batch_size: int = 16
    shuffle: bool = True
    epochs: int = 10
    loss_history: ClassVar[list[float]] = []
    verbose: bool = False
    preprocessor: Preprocessor | None = None
    loss_fn: nn.Module = nn.MSELoss()
    optimizer_cls: type[optim.Optimizer] = optim.Adam
    optimizer: optim.Optimizer
    lr: float = 1e-1
    scheduler_cls: type[LRScheduler] | None = None

    def preprocess(self, x: TensorLike) -> TensorLike:
        if self.preprocessor is None:
            return x
        return self.preprocessor.preprocess(x)

    def loss_func(self, y_pred, y_true):
        """
        Loss function to be used for training the model.
        This can be overridden by subclasses to use a different loss function.
        """
        return nn.MSELoss()(y_pred, y_true)

    @classmethod
    def scheduler_config(cls) -> dict:
        """
        Returns a random configuration for the learning rate scheduler.
        This should be added to the `get_tune_config()` method of subclasses
        to allow tuning of the scheduler parameters.
        """
        all_params = [
            {
                "scheduler_cls": [ExponentialLR],
                "scheduler_kwargs": [
                    {"gamma": 0.9},
                    {"gamma": 0.95},
                ],
            },
            # TODO: investigate these suggestions from copilot
            # {
            #     "scheduler_cls": [CosineAnnealingLR],
            #     "scheduler_kwargs": [{"T_max": 10, "eta_min": 0.01}],
            # },
            # {
            #     "scheduler_cls": [ReduceLROnPlateau],
            #     "scheduler_kwargs": [{"mode": "min", "factor": 0.1, "patience": 5}],
            # },
            # {
            #     "scheduler_cls": [StepLR],
            #     "scheduler_kwargs": [{"step_size": 10, "gamma": 0.1}],
            # },
            # {
            #     "scheduler_cls": [CyclicLR],
            #     "scheduler_kwargs": [{
            #         "base_lr": 1e-3,
            #         "max_lr": 1e-1,
            #         "step_size_up": 5,
            #         "step_size_down": 5,
            #     }],
            # },
            # {
            #     "scheduler_cls": [OneCycleLR],
            #     "scheduler_kwargs": [{
            #         "max_lr": 1e-1,
            #         "total_steps": self.epochs,
            #         "pct_start": 0.3,
            #         "anneal_strategy": "linear",
            #     }],
            # },
        ]
        # Randomly select one of the parameter sets
        return random.choice(all_params)

    def _fit(
        self,
        x: TensorLike,
        y: TensorLike,
    ):
        """
        Train a PyTorchBackend model.

        Parameters
        ----------
            X: TensorLike
                Input features as numpy array, PyTorch tensor, or DataLoader.
            y: OutputLike or None
                Target values (not needed if x is a DataLoader).

        """

        self.train()  # Set model to training mode

        # Convert input to DataLoader if not already
        dataloader = self._convert_to_dataloader(
            x, y, batch_size=self.batch_size, shuffle=self.shuffle
        )

        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches = 0

            for X_batch, y_batch in dataloader:
                # Preprocess x_batch
                x = self.preprocess(X_batch)

                # Forward pass
                y_pred = self.forward(X_batch)
                loss = self.loss_func(y_pred, y_batch)

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track loss
                epoch_loss += loss.item()
                batches += 1
            # Update learning rate if scheduler is defined
            if self.scheduler is not None:
                self.scheduler.step()  # type: ignore[call-arg]

            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / batches
            self.loss_history.append(avg_epoch_loss)

            if self.verbose and (epoch + 1) % (self.epochs // 10 or 1) == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}")

    def _initialize_weights(
        self,
        weight_init: str = "default",
        scale: float = 1.0,
        bias_init: str = "default",
    ):
        """Initialize the weights.

        Parameters
        ----------
        weight_init : str
            Initialization method name
        scale : float
            Scale parameter for initialization methods. Used as:
            - gain for Xavier methods
            - std for normal distribution
            - bound for uniform distribution (range: [-scale, scale])
            - ignored for Kaiming methods (uses optimal scaling)
        bias_init : str
            Bias initialization method. Options: "zeros", "default"
            "zeros" initializes biases to zero
            "default" uses PyTorch's default uniform initialization
        """
        # Dictionary mapping for weight initialization methods
        init_methods = {
            "xavier_uniform": lambda w: nn.init.xavier_uniform_(w, gain=scale),
            "xavier_normal": lambda w: nn.init.xavier_normal_(w, gain=scale),
            "kaiming_uniform": lambda w: nn.init.kaiming_uniform_(
                w, mode="fan_in", nonlinearity="relu"
            ),
            "kaiming_normal": lambda w: nn.init.kaiming_normal_(
                w, mode="fan_in", nonlinearity="relu"
            ),
            "normal": lambda w: nn.init.normal_(w, mean=0.0, std=scale),
            "uniform": lambda w: nn.init.uniform_(w, -scale, scale),
            "zeros": lambda w: nn.init.zeros_(w),
            "ones": lambda w: nn.init.ones_(w),
        }

        for module in self.modules():
            # TODO: consider and add handling for other module types
            if isinstance(module, nn.Linear):
                # Apply initialization if method exists
                if weight_init in init_methods:
                    init_methods[weight_init](module.weight)

                # Initialize biases based on bias_init parameter
                if module.bias is not None and bias_init == "zeros":
                    nn.init.zeros_(module.bias)

    def _predict(self, x: TensorLike) -> OutputLike:
        self.eval()
        with torch.no_grad():
            x = self.preprocess(x)
            return self(x)


class SklearnBackend(DeterministicEmulator):
    """
    SklearnBackend is a sklearn model and implements the base class.
    This provides default implementations to further subclasses.
    This means that models can subclass and only need to implement
    `.fit()` and `.predict()` to have an emulator to be run in `AutoEmulate`
    """

    model: BaseEstimator
    normalize_y: bool = False
    y_mean: TensorLike
    y_std: TensorLike

    def _model_specific_check(self, x: NumpyLike, y: NumpyLike):
        _, _ = x, y

    def _fit(self, x: TensorLike, y: TensorLike):
        if self.normalize_y:
            y, y_mean, y_std = self._normalize(y)
            self.y_mean = y_mean
            self.y_std = y_std
        x_np, y_np = self._convert_to_numpy(x, y)
        assert isinstance(x_np, np.ndarray)
        assert isinstance(y_np, np.ndarray)
        self.n_features_in_ = x_np.shape[1]
        self._model_specific_check(x_np, y_np)
        self.model.fit(x_np, y_np)  # type: ignore PGH003

    def _predict(self, x: TensorLike) -> TensorLike:
        x_np, _ = self._convert_to_numpy(x, None)
        y_pred = self.model.predict(x_np)  # type: ignore PGH003
        _, y_pred = self._move_tensors_to_device(*self._convert_to_tensors(x, y_pred))
        if self.normalize_y:
            y_pred = self._denormalize(y_pred, self.y_mean, self.y_std)
        return y_pred


class DropoutTorchBackend(PyTorchBackend):
    """
    Torch backend model that is meant to support dropout.
    """
