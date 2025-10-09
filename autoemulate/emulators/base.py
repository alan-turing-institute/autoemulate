import random
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch import nn, optim
from torch.distributions import TransformedDistribution
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import (
    DistributionLike,
    GaussianLike,
    NumpyLike,
    OutputLike,
    TensorLike,
    TuneParams,
)
from autoemulate.data.utils import ConversionMixin, ValidationMixin
from autoemulate.transforms.standardize import StandardizeTransform


class Emulator(ABC, ValidationMixin, ConversionMixin, TorchDeviceMixin):
    """
    Base class for all emulators.

    This class provides the basic structure and methods for emulators in AutoEmulate.
    It includes methods for fitting, predicting, and handling device management.

    """

    is_fitted_: bool = False
    supports_grad: bool = False
    scheduler_cls: type[optim.lr_scheduler.LRScheduler] | None = None
    x_transform: StandardizeTransform | None = None
    y_transform: StandardizeTransform | None = None
    supports_uq: bool = False

    @abstractmethod
    def _fit(self, x: TensorLike, y: TensorLike): ...

    def fit(self, x: TensorLike, y: TensorLike):
        """Fit the emulator to the provided data."""
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
        """Return the full name of the model."""
        return cls.__name__

    @classmethod
    def short_name(cls) -> str:
        """
        Return a short name for the model.

        Take the capital letters of the class name and return them as a lower case
        string. For example, if the class name is `GaussianProcess`, this will return
        `gp`.
        """
        return "".join([c for c in cls.__name__ if c.isupper()]).lower()

    @abstractmethod
    def _predict(self, x: TensorLike, with_grad: bool) -> OutputLike:
        pass

    def predict(self, x: TensorLike, with_grad: bool = False) -> OutputLike:
        """Predict the output for the given input.

        Parameters
        ----------
        x: TensorLike
            Input tensor to make predictions for.
        with_grad: bool
            Whether to enable gradient calculation. Defaults to False.

        Returns
        -------
        OutputLike
            The predicted output.
        """
        if not self.is_fitted_:
            msg = "Model is not fitted yet. Call fit() before predict()."
            raise RuntimeError(msg)
        self._check(x, None)
        x = self._ensure_with_grad(x, with_grad)
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

    def predict_mean(
        self, x: TensorLike, with_grad: bool = False, n_samples: int = 100
    ) -> TensorLike:
        """
        Predict the mean of the target variable for input `x`.

        Parameters
        ----------
        x: TensorLike
            Input tensor of shape `(n_batch, n_features)` for which to predict
            the mean.
        with_grad: bool
            Whether to compute gradients with respect to the input. Defaults to False.
        n_samples: int
            Number of samples to draw when using sampling-based predictions.
            Defaults to 100.

        Returns
        -------
        TensorLike
            Mean tensor of shape `(n_batch, n_targets)`.
        """
        x = self._ensure_with_grad(x, with_grad)
        y_pred = self._predict(x, with_grad)
        if isinstance(y_pred, TensorLike):
            return y_pred
        try:
            return y_pred.mean
        except Exception:
            # Use sampling to get a mean if mean property not available
            samples = (
                y_pred.rsample(torch.Size([n_samples]))
                if with_grad
                else y_pred.sample(torch.Size([n_samples]))
            )
            return samples.mean(dim=0)

    def predict_mean_and_variance(
        self, x: TensorLike, with_grad: bool = False, n_samples: int = 100
    ) -> tuple[TensorLike, TensorLike | None]:
        """
        Predict the mean and variance of the target variable for input `x`.

        Parameters
        ----------
        x: TensorLike
            Input tensor of shape `(n_batch, n_features)` for which to predict
            the mean and variance.
        with_grad: bool
            Whether to compute gradients with respect to the input. Defaults to False.
        n_samples: int
            Number of samples to draw when using sampling-based predictions.
            Defaults to 100.

        Returns
        -------
        tuple[TensorLike, TensorLike | None]
            A tuple containing:
            - Mean tensor of shape `(n_batch, n_targets)`.
            - Variance tensor of shape `(n_batch, n_targets)` if model supports UQ
            otherwise None.
        """
        x = self._ensure_with_grad(x, with_grad)
        if not self.supports_uq:
            return (self.predict_mean(x, with_grad, n_samples), None)
        y_pred = self._predict(x, with_grad)
        assert isinstance(y_pred, DistributionLike)
        try:
            return (y_pred.mean, y_pred.variance)
        except Exception:
            samples = (
                y_pred.rsample(torch.Size([n_samples]))
                if with_grad
                else y_pred.sample(torch.Size([n_samples]))
            )
            return samples.mean(dim=0), samples.var(dim=0)

    @staticmethod
    def _ensure_with_grad(x: TensorLike, with_grad: bool) -> TensorLike:
        """Ensure that the tensor x has requires_grad=True if with_grad is True.

        Parameters
        ----------
        x: TensorLike
            Input tensor.
        with_grad: bool
            Whether to enable gradient calculation.

        Returns
        -------
        TensorLike
            The input tensor with requires_grad set to True if with_grad is True.

        """
        if with_grad and isinstance(x, torch.Tensor) and not x.requires_grad:
            # Prefer enabling grad in-place on leaf tensors so callers can request
            # gradients w.r.t. their original input tensor.
            if x.is_leaf:
                x.requires_grad_(True)
            else:
                # Fall back to a detached leaf clone when we cannot mutate flags on
                # non-leaf tensors.
                x = x.clone().detach().requires_grad_(True)
        return x

    @staticmethod
    @abstractmethod
    def is_multioutput() -> bool:
        """Flag to indicate if the model is multioutput or not."""

    @staticmethod
    def get_tune_params() -> TuneParams:
        """
        Return a dictionary of hyperparameters to tune.

        The keys in the TuneParams must be implemented as keyword arguments in the
        __init__ method of any subclasses.

        e.g.

        tune_params: TuneParams = {
            "lr": list[0.01, 0.1],
            "batch_size": [16, 32],
            "mean"
        }

        model_params: ModelParams = {
            "lr": 0.01,
            "batch_size": 16
        }

        class MySubClass(Emulator):
            def __init__(lr, batch_size):
                self.lr = lr
                self.batch_size = batch_size
        """
        msg = (
            "Subclasses should implement for generating tuning params specific to "
            "each subclass."
        )
        raise NotImplementedError(msg)

    @classmethod
    def get_random_params(cls):
        """Return a random set of params for the model."""
        return {
            k: v[np.random.randint(len(v))] for k, v in cls.get_tune_params().items()
        }

    @classmethod
    def scheduler_params(cls) -> dict:
        """
        Return a random parameters for the learning rate scheduler.

        This should be added to the `get_tune_params()` method of subclasses
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

    def scheduler_setup(self, scheduler_kwargs: dict | None = None):
        """
        Set up the learning rate scheduler for the emulator.

        Parameters
        ----------
        scheduler_kwargs : dict | None
            Keyword arguments for the scheduler.
        """
        if scheduler_kwargs is None:
            msg = "Provide scheduler_kwargs to set up the scheduler."
            raise ValueError(msg)

        if not hasattr(self, "optimizer"):
            msg = "Optimizer must be set before setting up the scheduler."
            raise RuntimeError(msg)

        # Set up the scheduler if a scheduler class is defined
        if self.scheduler_cls is None:
            self.scheduler = None
        else:
            self.scheduler = self.scheduler_cls(self.optimizer, **scheduler_kwargs)  # type: ignore[call-arg]


class DeterministicEmulator(Emulator):
    """A base class for deterministic emulators."""

    supports_uq: bool = False

    @abstractmethod
    def _predict(self, x: TensorLike, with_grad: bool) -> TensorLike: ...
    def predict(self, x: TensorLike, with_grad: bool = False) -> TensorLike:
        """Predict the output for the given input.

        Parameters
        ----------
        x: TensorLike
            Input tensor to make predictions for.
        with_grad: bool
            Whether to enable gradient calculation. Defaults to False.

        Returns
        -------
        TensorLike
            The emulator predicted output for `x`.
        """
        pred = super().predict(x, with_grad)
        assert isinstance(pred, TensorLike)
        return pred

    def predict_mean_and_variance(
        self, x, with_grad=False, n_samples=100
    ) -> tuple[TensorLike, None]:
        """
        Predict the mean and variance of the target variable for input `x`.

        Parameters
        ----------
        x: TensorLike
            Input tensor of shape `(n_batch, n_features)` for which to predict
            the mean and variance.
        with_grad: bool
            Whether to compute gradients with respect to the input. Defaults to False.
        n_samples: int
            Number of samples to draw when using sampling-based predictions.
            Defaults to 100.

        Returns
        -------
        tuple[TensorLike, None]
            A tuple containing:
            - Mean tensor of shape `(n_batch, n_targets)`.
            - Variance tensor as `None` since the model does not support UQ.
        """
        mean, variance = Emulator.predict_mean_and_variance(
            self, x, with_grad, n_samples
        )
        assert variance is None
        return mean, variance


class ProbabilisticEmulator(Emulator):
    """A base class for probabilistic emulators."""

    supports_uq: bool = True

    @abstractmethod
    def _predict(self, x: TensorLike, with_grad: bool) -> DistributionLike: ...
    def predict(self, x: TensorLike, with_grad: bool = False) -> DistributionLike:
        """Predict the output distribution for the given input.

        Parameters
        ----------
        x: TensorLike
            Input tensor to make predictions for.
        with_grad: bool
            Whether to enable gradient calculation. Defaults to False.

        Returns
        -------
        DistributionLike
            The emulator predicted distribution for `x`.
        """
        pred = super().predict(x, with_grad)
        assert isinstance(pred, DistributionLike)
        return pred

    def predict_mean_and_variance(
        self, x, with_grad=False, n_samples=100
    ) -> tuple[TensorLike, TensorLike]:
        """
        Predict the mean and variance of the target variable for input `x`.

        Parameters
        ----------
        x: TensorLike
            Input tensor of shape `(n_batch, n_features)` for which to predict
            the mean and variance.
        with_grad: bool
            Whether to compute gradients with respect to the input. Defaults to False.
        n_samples: int
            Number of samples to draw when using sampling-based predictions.
            Defaults to 100.

        Returns
        -------
        tuple[TensorLike, None]
            A tuple containing:
            - Mean tensor of shape `(n_batch, n_targets)`.
            - Variance tensor of shape `(n_batch, n_targets)`.
        """
        mean, variance = Emulator.predict_mean_and_variance(
            self, x, with_grad, n_samples
        )
        assert isinstance(variance, TensorLike)
        return mean, variance


class GaussianEmulator(ProbabilisticEmulator):
    """A base class for Gaussian emulators."""

    supports_grad: bool = True

    @abstractmethod
    def _predict(self, x: TensorLike, with_grad: bool) -> GaussianLike: ...
    def predict(self, x: TensorLike, with_grad: bool = False) -> GaussianLike:
        """Predict the Gaussian distribution for the given input.

        Parameters
        ----------
        x: TensorLike
            Input tensor to make predictions for.
        with_grad: bool
            Whether to enable gradient calculation. Defaults to False.

        Returns
        -------
        GaussianLike
            The emulator predicted Gaussian distribution for `x`.
        """
        pred = super().predict(x, with_grad)
        assert isinstance(pred, GaussianLike)
        return pred


class GaussianProcessEmulator(GaussianEmulator):
    """A base class for Gaussian Process emulators."""

    @abstractmethod
    def _predict(self, x: TensorLike, with_grad: bool) -> GaussianLike: ...
    def predict(self, x: TensorLike, with_grad: bool = False) -> GaussianLike:
        """Predict the Gaussian distribution for the given input.

        Parameters
        ----------
        x: TensorLike
            Input tensor to make predictions for.
        with_grad: bool
            Whether to enable gradient calculation. Defaults to False.

        Returns
        -------
        GaussianLike
            The emulator predicted Gaussian distribution for `x`.
        """
        pred = super().predict(x, with_grad)
        assert isinstance(pred, GaussianLike)
        return pred


class PyTorchBackend(nn.Module, Emulator):
    """
    `PyTorchBackend` provides a backend for PyTorch models.

    The class provides the basic structure and methods for PyTorch-based emulators to
    enable further subclassing and customization. This provides default implementations
    to simplify model-specific subclasses by only needing to implement:
    - `.__init__()`: the constructor for the model
    - `.forward()`: the forward pass of the model
    - `.get_tune_params()`: the hyperparameters to tune for the model

    """

    batch_size: int = 16
    shuffle: bool = True
    epochs: int = 10
    loss_history: ClassVar[list[float]] = []
    verbose: bool = False
    loss_fn: nn.Module = nn.MSELoss()
    optimizer_cls: type[optim.Optimizer] = optim.Adam
    optimizer: optim.Optimizer
    supports_grad: bool = True
    lr: float = 1e-1
    scheduler_cls: type[LRScheduler] | None = None
    supports_uq: bool = False

    def loss_func(self, y_pred, y_true):
        """Loss function to be used for training the model."""
        return nn.MSELoss()(y_pred, y_true)

    def _fit(self, x: TensorLike, y: TensorLike):
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
            return self(x)


class SklearnBackend(DeterministicEmulator):
    """
    `SklearnBackend` provides a backend for sklearn models.

    The class provides the basic structure and methods for sklearn-based emulators to
    enable further subclassing and customization. This provides default implementations
    to simplify model-specific subclasses by only needing to implement:
    - `.__init__()`: the constructor for the model
    - `.get_tune_params()`: the hyperparameters to tune for the model
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
    """PyTorch backend model that is able to support dropout."""
