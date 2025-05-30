import random
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch import nn, optim

from autoemulate.experimental.data.preprocessors import Preprocessor
from autoemulate.experimental.data.utils import (
    ConversionMixin,
    ValidationMixin,
)
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.types import NumpyLike, OutputLike, TensorLike, TuneConfig


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

    @staticmethod
    def set_random_seed(seed: int, deterministic: bool = False):
        """Set random seed for Python, NumPy and PyTorch.

        Parameters
        ----------
        seed : int
            The random seed to use.
        deterministic : bool
            Use "deterministic" algorithms in PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)


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
    optimizer: optim.Optimizer

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

            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / batches
            self.loss_history.append(avg_epoch_loss)

            if self.verbose and (epoch + 1) % (self.epochs // 10 or 1) == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}")

    def _predict(self, x: TensorLike) -> OutputLike:
        self.eval()
        x = self.preprocess(x)
        return self(x)


class SklearnBackend(Emulator):
    """
    SklearnBackend is a sklearn model and implements the base class.
    This provides default implementations to further subclasses.
    This means that models can subclass and only need to implement
    `.fit()` and `.predict()` to have an emulator to be run in `AutoEmulate`
    """

    # TODO: consider if we also need to inherit from other classes
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

    def _predict(self, x: TensorLike) -> OutputLike:
        x_np, _ = self._convert_to_numpy(x, None)
        y_pred = self.model.predict(x_np)  # type: ignore PGH003
        _, y_pred = self._move_tensors_to_device(*self._convert_to_tensors(x, y_pred))
        if self.normalize_y:
            y_pred = self._denormalize(y_pred, self.y_mean, self.y_std)
        return y_pred
