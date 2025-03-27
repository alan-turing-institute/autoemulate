from abc import ABC
from abc import abstractmethod

import gpytorch
import numpy as np
import torch
import torch.optim as optim
from gpytorch.likelihoods.likelihood import Likelihood
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset


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
        ...

    @abstractmethod
    def predict(self, X: DataLoader):
        ...

    @abstractmethod
    def tune(self, X: Dataset):
        ...

    @abstractmethod
    def cross_validate(self, X: Dataset):
        ...


class InputTypeMixin:
    def convert(
        self,
        X: np.ndarray | torch.Tensor | DataLoader,
        y: np.ndarray | torch.Tensor | None = None,
        batch_size: int = 16,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Mixin class to convert input data to pytorch DataLoaders.
        """
        # Convert input to DataLoader if not already
        if isinstance(X, np.ndarray):
            x = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        if isinstance(X, (torch.Tensor, np.ndarray)) and y is not None:
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        elif isinstance(X, DataLoader) and y is None:
            dataloader = X
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


# Maybe come back later
# class CommonTorchBackend(nn.Module, BaseEmulator):
#     pass


class PyTorchBackend(nn.Module, BaseEmulator, InputTypeMixin):
    """PyTorchBackend is a torch model and implements the base class.
    This provides default implementations to further subclasses.
    This means that models can subclass and only need to implement
    `.forward()` to have an emulator to be run in `AutoEmulate`"""

    def fit(
        self,
        X: np.ndarray | torch.Tensor | DataLoader,
        y: np.ndarray | None | torch.Tensor = None,
        epochs: int = 3,
        batch_size: int = 1,
        shuffle: bool = True,
        verbose: bool = True,
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
        dataloader = self.convert(X, y, batch_size=batch_size, shuffle=shuffle)

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
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

        return loss_history

    def predict(self, X: np.ndarray | torch.Tensor | DataLoader) -> Tensor:
        self.eval()
        return self(X)

    def cross_validate(self, X):
        raise NotImplementedError("This function is not yet implemented.")

    def tune(self, X):
        raise NotImplementedError("This function is not yet implemented.")


class GPyTorchBackend(gpytorch.models.ExactGP, BaseEmulator, InputTypeMixin):
    # TODO: can the init method be different across models?
    likelihood: Likelihood
    mll: gpytorch.mlls.MarginalLogLikelihood

    # my_class.fit(X, y, my_ll(, my_class))
    # def fit(self, X: torch.Tensor | np.ndarray | DataLoader, y: torch.Tensor | np.ndarray, mll: gpytorch.mlls.ExactMarginalLogLikelihood):
    def fit(
        self, X: torch.Tensor | np.ndarray | DataLoader, y: torch.Tensor | np.ndarray
    ):
        # TODO: consider impl setting batch_size N when passed Tensor
        dataloader = self.convert(X, y, batch_size=X.shape[0])
        mll = self.mll(self.likelihood, self)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        self.train()
        self.likelihood.train()
        for i in range(self.max_epochs):
            # Since only batch size
            for X, y in dataloader:
                optimizer.zero_grad()
                output = self(X)
                loss = -mll(output, y)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        self.eval()
        return self(X)

    def tune(self, X, y):
        raise NotImplementedError("This function is not yet implemented.")

    def cross_validate(self, X, y):
        raise NotImplementedError("This function is not yet implemented.")


class GaussianProcessRBFExact(GPyTorchBackend):
    # Perhaps a match statement to handle the different cases will be suffiocient
    def __init__(self, train_x, train_y, normalize_y=True):
        X, y = train_x, train_y
        self.mean_module = None
        self.covar_module = None
        self.normalize_y = normalize_y
        self.n_features_in_ = X.shape[1]
        # Single task for now
        self.n_outputs_ = 1
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # TODO: check if the self need initialization?
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        # GP's work better when the target values are normalized
        if self.normalize_y:
            self._y_train_mean = y.mean(dim=0)
            self._y_train_std = y.std(dim=0)
            y = (y - self._y_train_mean) / self._y_train_std

        mean_module: gpytorch.means.Mean = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([self.n_outputs_])
        )

        # combined RBF + constant kernel works well in a lot of cases
        rbf = gpytorch.kernels.RBFKernel(
            ard_num_dims=self.n_features_in_,  # different lengthscale for each feature
            batch_shape=torch.Size([self.n_outputs_]),  # batched multioutput
            # seems to work better when we initialize the lengthscale
        ).initialize(lengthscale=torch.ones(self.n_features_in_) * 1.5)
        constant = gpytorch.kernels.ConstantKernel()
        combined = rbf + constant
        covar_module = gpytorch.kernels.ScaleKernel(
            combined, batch_shape=torch.Size([self.n_outputs_])
        )
        super(GPyTorchBackend, self).__init__(X, y, self.likelihood)
        self.max_epochs = 100
        self.mean_module = mean_module
        self.covar_module: gpytorch.kernels.Kernel = covar_module

    def forward(
        self, x
    ) -> (
        gpytorch.distributions.Distribution
        | torch.distributions.Distribution
        | torch.Tensor
    ):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
