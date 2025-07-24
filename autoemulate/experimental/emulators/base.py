import random
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch import nn, optim
from torch.distributions import TransformedDistribution
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler

from autoemulate.experimental.data.preprocessors import Preprocessor
from autoemulate.experimental.data.utils import ConversionMixin, ValidationMixin
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.transforms.standardize import StandardizeTransform
from autoemulate.experimental.types import (
    DistributionLike,
    GaussianLike,
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
    supports_grad: bool = False
    scheduler_cls: type[optim.lr_scheduler.LRScheduler] | None = None
    x_transform: StandardizeTransform | None = None
    y_transform: StandardizeTransform | None = None

    @abstractmethod
    def _fit(self, x: TensorLike, y: TensorLike): ...

    def fit(self, x: TensorLike, y: TensorLike):
        self._check(x, y)
        # Ensure x and y are tensors and 2D
        x, y = self._convert_to_tensors(x, y)

        # Move to device
        x, y = self._move_tensors_to_device(x, y)

        # Fit transforms
        if self.x_transform is not None:
            self.x_transform.fit(x)
        if self.y_transform is not None:
            self.y_transform.fit(y)
        x = self.x_transform(x) if self.x_transform is not None else x
        y = self.y_transform(y) if self.y_transform is not None else y

        # Fit emulator
        self._fit(x, y)
        self.is_fitted_ = True

    @abstractmethod
    def __init__(
        self, x: TensorLike | None = None, y: TensorLike | None = None, **kwargs
    ): ...

    @classmethod
    def model_name(cls) -> str:
        return cls.__name__

    @classmethod
    def short_name(cls) -> str:
        """
        Take the capital letters of the class name and return them as a lower case
        string. For example, if the class name is `GaussianProcessExact`, this will
        return `gpe`.
        """
        return "".join([c for c in cls.__name__ if c.isupper()]).lower()

    @abstractmethod
    def _predict(self, x: TensorLike, with_grad: bool) -> OutputLike:
        pass

    def predict(self, x: TensorLike, with_grad: bool = False) -> OutputLike:
        if not self.is_fitted_:
            msg = "Model is not fitted yet. Call fit() before predict()."
            raise RuntimeError(msg)
        self._check(x, None)
        (x,) = self._move_tensors_to_device(x)
        x = self.x_transform(x) if self.x_transform is not None else x
        output = self._predict(x, with_grad)
        if self.y_transform is not None:
            if isinstance(output, GaussianLike):
                output = self.y_transform._inverse_gaussian(output)
            elif isinstance(output, DistributionLike):
                output = TransformedDistribution(
                    output, transforms=[self.y_transform.inv]
                )
            elif isinstance(output, TensorLike):
                output = self.y_transform.inv(output)
            else:
                msg = (
                    "Output type not supported for transformation. "
                    f"Got {type(output)} but expected GaussianLike, "
                    "DistributionLike, or TensorLike."
                )
                raise TypeError(msg)
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

    @classmethod
    def scheduler_config(cls) -> dict:
        """
        Returns a random configuration for the learning rate scheduler.
        This should be added to the `get_tune_config()` method of subclasses
        to allow tuning of the scheduler parameters.
        """
        all_params = [
            {
                "scheduler_cls": [None],
                "scheduler_kwargs": [{}],
            },
            {
                "scheduler_cls": [ExponentialLR],
                "scheduler_kwargs": [
                    {"gamma": 0.9},
                    {"gamma": 0.95},
                ],
            },
            {
                "scheduler_cls": [LRScheduler],
                "scheduler_kwargs": [
                    {"policy": "ReduceLROnPlateau", "patience": 5, "factor": 0.5}
                ],
            },
            # TODO: investigate these suggestions from copilot, issue: #597
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

    def scheduler_setup(self, kwargs: dict | None = None):
        """
        Setup the learning rate scheduler for the emulator.
        Parameters
        ----------
        kwargs : dict | None
            Keyword arguments for the model. This should include scheduler_kwargs.
        """
        if kwargs is None:
            msg = (
                "Provide a kwargs dictionary including "
                "scheduler_kwargs to set up the scheduler."
            )
            raise ValueError(msg)

        if not hasattr(self, "optimizer"):
            msg = "Optimizer must be set before setting up the scheduler."
            raise RuntimeError(msg)

        # Extract scheduler-specific kwargs if present
        try:
            assert type(kwargs) is dict
            scheduler_kwargs = kwargs.pop("scheduler_kwargs", {})
        except AttributeError:
            # If kwargs does not contain scheduler_kwargs, throw an error
            msg = "No kwargs for scheduler setup detected."
            raise ValueError(msg) from None

        # Set up the scheduler if a scheduler class is defined
        if self.scheduler_cls is None:
            self.scheduler = None
        else:
            self.scheduler = self.scheduler_cls(self.optimizer, **scheduler_kwargs)  # type: ignore[call-arg]


class DeterministicEmulator(Emulator):
    """An emulator subclass that predicts with deterministic outputs returning a
    `TensorLike`.
    """

    @abstractmethod
    def _predict(self, x: TensorLike, with_grad: bool) -> TensorLike: ...
    def predict(self, x: TensorLike, with_grad: bool = False) -> TensorLike:
        pred = super().predict(x, with_grad)
        assert isinstance(pred, TensorLike)
        return pred


class ProbabilisticEmulator(Emulator):
    """
    An emulator subclass that predicts with probabilistic outputs returning a
    `DistributionLike`.
    """

    @abstractmethod
    def _predict(self, x: TensorLike, with_grad: bool) -> DistributionLike: ...
    def predict(self, x: TensorLike, with_grad: bool = False) -> DistributionLike:
        pred = super().predict(x, with_grad)
        assert isinstance(pred, DistributionLike)
        return pred

    def predict_mean_and_variance(
        self, x: TensorLike, with_grad: bool = False
    ) -> tuple[TensorLike, TensorLike]:
        """
        Predict mean and variance from the probabilistic output.

        Parameters
        ----------
        x: TensorLike
            Input tensor to make predictions for.
        with_grad: bool
            Whether to enable gradient calculation. Defaults to False.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            The emulator predicted mean and variance for `x`.
        """
        pred = self.predict(x, with_grad)
        return pred.mean, pred.variance


class GaussianEmulator(ProbabilisticEmulator):
    """
    An emulator subclass that predicts with Gaussian outputs returning a
    `GaussianLike`.
    """

    supports_grad: bool = True

    @abstractmethod
    def _predict(self, x: TensorLike, with_grad: bool) -> GaussianLike: ...
    def predict(self, x: TensorLike, with_grad: bool = False) -> GaussianLike:
        pred = super().predict(x, with_grad)
        assert isinstance(pred, GaussianLike)
        return pred


class GaussianProcessEmulator(GaussianEmulator):
    """
    A Gaussian Process emulator subclass that predicts with output
    `GaussianProcessLike`.
    """

    @abstractmethod
    def _predict(self, x: TensorLike, with_grad: bool) -> GaussianLike: ...
    def predict(self, x: TensorLike, with_grad: bool = False) -> GaussianLike:
        pred = super().predict(x, with_grad)
        assert isinstance(pred, GaussianLike)
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
    supports_grad: bool = True
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

    def _fit(
        self,
        x: TensorLike,
        y: TensorLike,
    ):
        """
        Train a PyTorchBackend model.

        Parameters
        ----------
        x: TensorLike
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

            for x_batch, y_batch in dataloader:
                # Preprocess x_batch
                x = self.preprocess(x_batch)

                # Forward pass
                y_pred = self.forward(x_batch)
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
        """
        Initialize the weights.

        Parameters
        ----------
        weight_init: str
            Initialization method name
        scale: float
            Scale parameter for initialization methods. Used as:
            - gain for Xavier methods
            - std for normal distribution
            - bound for uniform distribution (range: [-scale, scale])
            - ignored for Kaiming methods (uses optimal scaling)
        bias_init: str
            Bias initialization method. Options: "zeros", "default":
                - "zeros" initializes biases to zero
                - "default" uses PyTorch's default uniform initialization
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

    def _predict(self, x: TensorLike, with_grad: bool) -> OutputLike:
        self.eval()
        with torch.set_grad_enabled(with_grad):
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
    supports_grad: bool = False

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

    def _predict(self, x: TensorLike, with_grad: bool) -> TensorLike:
        if with_grad:
            msg = "Gradient calculation is not supported."
            raise ValueError(msg)
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
