from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autoemulate.experimental.config import FitConfig
from autoemulate.experimental.types import InputLike, OutputLike

_default_fit_config = FitConfig(
    epochs=10,
    batch_size=16,
    shuffle=True,
    criterion=torch.nn.MSELoss,
    optimizer=torch.optim.Adam,
    device="cpu",
    verbose=False,
)


class Emulator(ABC):
    """The interface containing methods on emulators that are
    expected by downstream dependents. This includes:
    - `AutoEmulate`
    - Active learning implementations
    """

    # TODO: currently this has an issue with recursion
    # def __call__(self, *args, **kwds):
    #     return self.predict(*args, **kwds)

    @abstractmethod
    def fit(
        self,
        x: InputLike,
        y: OutputLike | None,
        config: FitConfig,
    ): ...

    @abstractmethod
    def predict(self, x: InputLike) -> OutputLike:
        pass

    @abstractmethod
    def tune(self, x: InputLike): ...

    @abstractmethod
    def cross_validate(self, x: InputLike): ...


class InputTypeMixin:
    def _convert(
        self,
        x: InputLike,
        y: OutputLike | None = None,
        batch_size: int = 16,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Mixin class to convert input data to pytorch DataLoaders.
        """
        # Convert input to DataLoader if not already
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        if isinstance(x, (torch.Tensor, np.ndarray)) and isinstance(
            y, (torch.Tensor, np.ndarray)
        ):
            dataset = TensorDataset(x, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        elif isinstance(x, DataLoader) and y is None:
            dataloader = x
        else:
            raise ValueError(
                "Unsupported type for X. Must be numpy array, PyTorch tensor, or DataLoader."
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

    loss_history: list[float]

    def fit(
        self,
        x: InputLike,
        y: OutputLike | None = None,
        config: FitConfig = _default_fit_config,
    ):
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
        if not isinstance(config, FitConfig):
            raise ValueError("config must be an instance of FitConfig")

        self.train()  # Set model to training mode

        # Convert input to DataLoader if not already
        dataloader = self._convert(
            x, y, batch_size=config.batch_size, shuffle=config.shuffle
        )

        # Training loop
        for epoch in range(config.epochs):
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
            self.loss_history.append(avg_epoch_loss)

            if config.verbose and (epoch + 1) % (config.epochs // 10 or 1) == 0:
                print(
                    f"Epoch [{epoch + 1}/{config.epochs}], Loss: {avg_epoch_loss:.4f}"
                )

    def predict(self, x: InputLike) -> OutputLike:
        self.eval()
        return self(x)

    def cross_validate(self, x: InputLike) -> None:
        raise NotImplementedError("This function is not yet implemented.")

    def tune(self, x: InputLike) -> None:
        raise NotImplementedError("This function is not yet implemented.")
