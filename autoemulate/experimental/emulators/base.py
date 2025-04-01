from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from autoemulate.experimental.types import (
    InputLike,
    OutputLike,
    TuneConfig,
)


class Emulator(ABC):
    """The interface containing methods on emulators that are
    expected by downstream dependents. This includes:
    - `AutoEmulate`
    """

    def __call__(self, *args, **kwds):
        return self.predict(*args, **kwds)

    @abstractmethod
    def fit(self, x: InputLike, y: OutputLike | None): ...

    @abstractmethod
    def predict(self, x: InputLike) -> OutputLike:
        pass

    @staticmethod
    @abstractmethod
    def get_tune_config() -> TuneConfig: ...

    @abstractmethod
    def cross_validate(self, x: InputLike): ...


class InputTypeMixin:
    def _convert(
        self,
        x: InputLike,
        y: OutputLike | None = None,
        batch_size: int = 16,
        shuffle: bool = True,
    ) -> DataLoader | Dataset:
        """
        Mixin class to convert input data to pytorch DataLoaders.
        """
        # Convert input to DataLoader if not already
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        if isinstance(x, (torch.Tensor, np.ndarray)) and y is not None:
            dataset = TensorDataset(x, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        elif isinstance(x, DataLoader) and y is None:
            dataloader = x
        elif isinstance(x, Dataset) and y is None:
            dataloader = x
        else:
            raise ValueError(
                f"Unsupported type for X ({type(x)}). Must be numpy array, PyTorch tensor, or DataLoader."
            )

        return dataloader

    # TODO: consider possible method for predict
    # def convert_x(self, y: np.ndarray | torch.Tensor | Data) -> torch.Tensor:
    #     if isinstance(y, np.ndarray):
    #         y = torch.tensor(y, dtype=torch.float32)
    #     else:
    #         raise ValueError("Unsupported type for X. Must be numpy array, PyTorch tensor")
    #     return y


class PyTorchBackend(nn.Module, Emulator, InputTypeMixin):
    """PyTorchBackend is a torch model and implements the base class.
    This provides default implementations to further subclasses.
    This means that models can subclass and only need to implement
    `.forward()` to have an emulator to be run in `AutoEmulate`"""

    batch_size: int = 16
    shuffle: bool = True
    epochs: int = 10
    verbose: bool = False

    def fit(
        self,
        x: InputLike,
        y: OutputLike | None,
    ) -> list:
        """
        Train the linear regression model.

        Args:
            X: Input features as numpy array, PyTorch tensor, or DataLoader
            y: Target values (not needed if xis a DataLoader)
            epochs: Number of training epochs
            batch_size: Batch size (used only when xis not a DataLoader)
            verbose: Whether to print progress

        Returns:
            List of loss values per epoch
        """

        self.train()  # Set model to training mode
        loss_history = []

        # Convert input to DataLoader if not already
        dataloader = self._convert(
            x, y, batch_size=self.batch_size, shuffle=self.shuffle
        )

        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches = 0

            for X_batch, y_batch in dataloader:
                # Forward pass
                y_pred = self.forward(X_batch)
                loss = self.loss_fn(y_pred, y_batch)

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track loss
                epoch_loss += loss.item()
                batches += 1

            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / batches
            loss_history.append(avg_epoch_loss)

            if self.verbose and (epoch + 1) % (self.epochs // 10 or 1) == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}")

        return loss_history

    def predict(self, x: InputLike) -> OutputLike:
        self.eval()
        return self(x)

    def cross_validate(self, x: InputLike) -> None:
        raise NotImplementedError("This function is not yet implemented.")

    @staticmethod
    def get_tune_config():
        return {
            "epochs": [100, 200, 300],
            "batch_size": [16, 32],
            "hidden_dim": [32, 64, 128],
            "latent_dim": [32, 64, 128],
            "max_context_points": [5, 10, 15],
            "hidden_layers_enc": [2, 3, 4],
            "hidden_layers_dec": [2, 3, 4],
            "activation": [
                nn.ReLU,
                nn.GELU,
            ],
            "optimizer": [torch.optim.Adam],
            "lr": list(np.logspace(-6, -1)),
        }
