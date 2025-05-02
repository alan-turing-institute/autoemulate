from abc import ABC, abstractmethod
from typing import ClassVar

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch import Tensor, nn, optim

from autoemulate.experimental.data.preprocessors import Preprocessor
from autoemulate.experimental.data.utils import InputTypeMixin
from autoemulate.experimental.data.validation import Base
from autoemulate.experimental.types import InputLike, OutputLike, TuneConfig


class Emulator(ABC, Base, InputTypeMixin):
    """
    The interface containing methods on emulators that are
    expected by downstream dependents. This includes:
    - `AutoEmulate`
    """

    # TODO: update emulators with these methods
    # @abstractmethod
    # def _fit(self, x: InputLike, y: InputLike | None): ...

    # def fit(self, x: InputLike, y: InputLike | None):
    #     self._check(x, y)
    #     self._fit(x, y)

    @abstractmethod
    def fit(self, x: InputLike, y: InputLike | None): ...

    @abstractmethod
    def __init__(
        self, x: InputLike | None = None, y: InputLike | None = None, **kwargs
    ): ...

    @classmethod
    def model_name(cls) -> str:
        return cls.__name__

    # TODO: update emulators with these methods
    # @abstractmethod
    # def _predict(self, x: InputLike) -> OutputLike:
    #     pass

    # def predict(self, x: InputLike) -> OutputLike:
    #     self._check(x, None)
    #     output = self.predict(x)
    #     # Check that it is Gaussian or Y
    #     self._check_output(output)
    #     return output

    @abstractmethod
    def predict(self, x: InputLike) -> OutputLike:
        pass

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

    def preprocess(self, x):
        if self.preprocessor is None:
            return x
        return self.preprocessor.preprocess(x)

    def loss_func(self, y_pred, y_true):
        """
        Loss function to be used for training the model.
        This can be overridden by subclasses to use a different loss function.
        """
        return nn.MSELoss()(y_pred, y_true)

    def fit(
        self,
        x: InputLike,
        y: InputLike | None,
    ):
        """
        Train a PyTorchBackend model.

        Parameters
        ----------
            X: InputLike
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

    def predict(self, x: InputLike) -> OutputLike:
        self.eval()
        x = self.preprocess(x)
        return self(x)

    def cross_validate(self, x: InputLike) -> None:
        msg = "This function is not yet implemented."
        raise NotImplementedError(msg)

    @staticmethod
    def get_tune_config():
        msg = (
            "Subclasses should implement for generating tuning config specific to "
            "each subclass."
        )
        raise NotImplementedError(msg)


# class SklearnEstimator(BaseEstimator, RegressorMixin): ...

class SklearnBackend(Emulator, BaseEstimator, RegressorMixin):
    """
    SklearnBackend is a sklearn model and implements the base class.
    This provides default implementations to further subclasses.
    This means that models can subclass and only need to implement
    `.fit()` and `.predict()` to have an emulator to be run in `AutoEmulate`
    """

    # model: SklearnEstimator

    def __init__(
        self, x: InputLike | None = None, y: InputLike | None = None, **kwargs
    ):
        pass

    def check_and_convert(self, x: InputLike, y: InputLike | None):
        x, y = self._convert_to_numpy(x, y)
        # if y is None:
        #     msg = "y must be provided."
        #     raise ValueError(msg)
        # if y.ndim > 2:
        #     msg = f"y must be 1D or 2D array. Found {y.ndim}D array."
        #     raise ValueError(msg)
        # if y.ndim == 2:  # _convert_to_numpy may return 2D y
        #     y = y.ravel()  # Ensure y is 1-dimensional
        self.n_features_in_ = x.shape[1]

        return x, y

    def _fit(self, x: InputLike, y: InputLike | None):
        self.model.fit(x, y)
        self.is_fitted_ = True

    # def fit(self, x: InputLike, y: InputLike | None):
    #     """Fits the emulator to the data."""
    #     x, y = self.check_and_convert(x, y)
    #     self._fit(x, y)

    def _predict(self, x: InputLike) -> OutputLike:
        check_is_fitted(self)
        x = check_array(x)
        return self.model.predict(x)

    def predict(self, x: InputLike) -> OutputLike:
        """Predicts the output of the emulator for a given input."""
        y_pred = self._predict(x)

        # Ensure the output is a 2D tensor array with shape (n_samples, 1)
        return Tensor(y_pred.reshape(-1, 1))  # type: ignore PGH003

    @staticmethod
    def get_tune_config() -> TuneConfig:
        return {}

