from abc import ABC, abstractmethod
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Union, Tuple, Optional
from gpytorch.models import ExactGP
import gpytorch
from dataclasses import dataclass


class BaseEmulator(ABC):
    """The interface containing methods on emulators that are
    expected by downstream dependents. This includes:
    - `AutoEmulate`
    - Active learning implementations
    """
    def __call__(self, *args, **kwds):
        return self.predict(*args, **kwds)

    @abstractmethod
    def fit(self, X: DataLoader):
        pass

    @abstractmethod
    def predict(self, X: DataLoader):
        pass

    @abstractmethod
    def tune(self, X: Dataset):
        pass

    @abstractmethod
    def cross_validate(self, X: Dataset):
        pass


# Maybe come back later
# class CommonTorchBackend(nn.Module, BaseEmulator):
#     pass

class PyTorchBackend(nn.Module, BaseEmulator):
    """PyTorchBackend is a torch model and implements the base class.
    This provides default implementations to further subclasses.
    This means that models can subclass and only need to implement
    `.forward()` to have an emulator to be run in `AutoEmulate`"""

    def fit(self,
                   X: np.ndarray | torch.Tensor | DataLoader,
                   y: np.ndarray | None | torch.Tensor = None,
                   epochs: int = 3,
                   batch_size: int = 1,
                   verbose: bool = True) -> list:
        """
        Train the linear regression model.

        Args:
            X: Input features as numpy array, PyTorch tensor, or DataLoader
            y: Target values (not needed if X is a DataLoader)
            epochs: Number of training epochs
            batch_size: Batch size (used only when X is not a DataLoader)
            verbose: Whether to print progress

        Returns:
            List of loss values per epoch
        """
        self.train()  # Set model to training mode
        loss_history = []

        # Convert input to DataLoader if not already
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        if isinstance(X, (torch.Tensor, np.ndarray)) and y is not None:
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        elif isinstance(X, DataLoader):
            dataloader = X
        else:
            raise ValueError("Unsupported type for X. Must be numpy array, PyTorch tensor, or DataLoader.")


        # Training loop
        for epoch in range(epochs):
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

            if verbose and (epoch + 1) % (epochs // 10 or 1) == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

        return loss_history

    def predict(self, X: np.ndarray | torch.Tensor | DataLoader) -> Tensor:
        self.eval()
        return self(X)

    def cross_validate(self, X):
        pass

    def tune(self, X):
        pass

class LinearRegression(PyTorchBackend):
    """Inherits from the PyTorchBackend.
    To complete the model, `.forward()` needs to be implemented. Methods from
    `BaseEmulator` could be optionally overridden too.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(1, 1, bias=True))
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        # Complete the model by passing through the linear layer

        return self.linear(x)

class GpyTorchExact(BaseEmulator, ExactGP):

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def cross_validate(self, X):
        pass

    def tune(self, X):
        pass

class RBFKernelGuassian(GpyTorchExact):
    def __init__(self):
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.optimizer = torch.optim.Adam(lr=0.1)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.loss = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == "__main__":

    # Generate some random data
    np.random.seed(42)
    X = np.arange(0, 10).reshape(-1, 1).astype(np.float32)
    y = 2 * X + 1

    model = LinearRegression()
    model.fit(X, y, epochs=50, batch_size=1)

    # Predict
    x_pred = torch.tensor([[44.0]])
    y_pred = model.predict(x_pred)
    print(f'Input: {x_pred.item()}')
    print(f'Prediction: {y_pred.item()}')
    print(f'Actual: {2 * x_pred.item() + 1}')

