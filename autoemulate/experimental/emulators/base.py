import random
from abc import ABC, abstractmethod
from typing import ClassVar

import lightning as L
import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch import nn, optim

from autoemulate.experimental.data.preprocessors import Preprocessor
from autoemulate.experimental.data.utils import ConversionMixin, ValidationMixin
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.types import NumpyLike, OutputLike, TensorLike, TuneConfig


class Emulator(ABC, ValidationMixin, ConversionMixin, TorchDeviceMixin):
    # class Emulator(ABC, ValidationMixin, ConversionMixin):
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
        # x, y = self._move_tensors_to_device(x, y)
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


# class PyTorchBackend(nn.Module, Emulator, Preprocessor):
class PyTorchBackend(L.LightningModule, Emulator, Preprocessor):
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
    optimizer: type[optim.Optimizer]
    # optimizer: type[optim.Adam]

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
        trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
        train_dataloader, val_dataloader = self._random_split(
            self._convert_to_dataset(x, y), self.batch_size
        )
        trainer.fit(
            self,
            train_dataloader,
            val_dataloader,
        )

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
        # TODO:
        trainer = L.Trainer()
        predictions = trainer.predict(self, self._convert_to_dataloader(x))
        assert predictions is not None
        return predictions

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def training_step(self, batch, batch_idx) -> TensorLike:
        X_batch, y_batch = batch
        x = self.preprocess(X_batch)
        y_pred = self.forward(x)
        return self.loss_func(y_pred, y_batch)

    def validation_step(self, batch, batch_idx) -> TensorLike:
        X_batch, y_batch = batch
        x = self.preprocess(X_batch)
        y_pred = self.forward(x)
        val_loss = self.loss_func(y_pred, y_batch)
        self.log("val_loss", val_loss)

    # def predict_step(self, *args, **kwargs):
    #     return super().predict_step(*args, **kwargs)

    def configure_optimizers(self):
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
        """
        optimizer = self.optimizer(self.parameters())
        # if self.lr_scheduler is not None:
        #     lr_scheduler = self.lr_scheduler(optimizer)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
        #     }
        # else:
        # return {"optimizer": optimizer}
        return optimizer


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
