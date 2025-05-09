from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from torch import Tensor, nn, optim
from sklearn.utils.validation import check_X_y

from autoemulate.experimental.data.preprocessors import Preprocessor
from autoemulate.experimental.data.utils import InputTypeMixin
from autoemulate.experimental.data.validation import ValidationMixin
from autoemulate.experimental.types import OutputLike, TensorLike, TuneConfig


class Emulator(ABC, ValidationMixin, InputTypeMixin):
    """
    The interface containing methods on emulators that are
    expected by downstream dependents. This includes:
    - `AutoEmulate`
    """

    @abstractmethod
    def _fit(self, x: TensorLike, y: TensorLike): ...

    def fit(self, x: TensorLike, y: TensorLike):
        self._check(x, y)
        self._fit(x, y)

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
        self._check(x, None)
        output = self._predict(x)
        self._check_output(output)
        return output

    @staticmethod
    @abstractmethod
    def is_multioutput() -> bool:
        """Flag to indicate if the model is multioutput or not."""

    @staticmethod
    @abstractmethod
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

        ...


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
                # TODO: consider if this should be moved outside of dataloader iteration
                # e.g. as part of the InputTypeMixin
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

    @staticmethod
    def get_tune_config():
        msg = (
            "Subclasses should implement for generating tuning config specific to "
            "each subclass."
        )
        raise NotImplementedError(msg)


class SklearnBackend(Emulator):
    """
    SklearnBackend is a sklearn model and implements the base class.
    This provides default implementations to further subclasses.
    This means that models can subclass and only need to implement
    `.fit()` and `.predict()` to have an emulator to be run in `AutoEmulate`
    """

    # TODO: consider if we also need to inherit from other classes
    model: BaseEstimator

    def _fit(self, x: TensorLike, y: TensorLike):
        x_np, y_np = self._convert_to_numpy(x, y)
        assert isinstance(x_np, np.ndarray)
        assert isinstance(y_np, np.ndarray)
        self.n_features_in_ = x.shape[1]

        if (y_np.ndim == 2 and y_np.shape[1] == 1) or y_np.ndim == 1:
            y_np = y_np.ravel()  # Ensure y is 1-dimensional
            x_np, y_np = check_X_y(x_np, y_np, multi_output=False, y_numeric=True)
        else:
            x_np, y_np = check_X_y(x_np, y_np, multi_output=True, y_numeric=True)
        self.model.fit(x_np, y_np)  # type: ignore PGH003
        self.is_fitted_ = True

    def _predict(self, x: TensorLike) -> OutputLike:
        check_is_fitted(self)
        x = check_array(x)
        y_pred = self.model.predict(x)  # type: ignore PGH003
        return Tensor(y_pred.reshape(-1, 1))  # type: ignore PGH003

    @staticmethod
    def get_tune_config() -> TuneConfig:
        return {}
