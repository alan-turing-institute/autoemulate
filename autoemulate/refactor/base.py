import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
import numpy as np

from autoemulate.refactor.utils import convert_to_numpy, convert_to_tensor


class BaseEmulator(ABC):
    @abstractmethod
    def fit(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        # TODO: refine with config/callbacks?
        epochs=10,
        device="cpu",
    ):
        pass

    @abstractmethod
    def predict(self, dataloader: DataLoader, device: str = "cpu"):
        pass

    @abstractmethod
    def tune(
        self,
        dataset: Dataset,
        criterion: nn.Module,
        optimizer: Optimizer,
        # TODO: could be replaced by general config
        param_grid,
        k: int = 3,
        epochs: int = 10,
        batch_size: int = 32,
        device="cpu",
    ):
        pass

    @abstractmethod
    def cross_validate(
        self,
        dataset: Dataset,
        criterion: nn.Module,
        optimizer: Optimizer,
        # TODO: could be replaced by general config
        k=5,
        epochs=10,
        batch_size=32,
        device="cpu",
    ):
        pass


class PyTorchBackend(nn.Module, BaseEmulator):
    def __init__(self):
        super().__init__()

    def fit(self, dataloader, criterion, optimizer, epochs=10, device="cpu"):
        self.to(device)
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    def predict(self, dataloader, device="cpu"):
        self.to(device)
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(device)
                outputs = self(inputs)
                predictions.append(outputs.cpu())
        return torch.cat(predictions, dim=0)

    def cross_validate(
        self,
        dataset,
        criterion,
        optimizer,
        k=5,
        epochs=10,
        batch_size=32,
        device="cpu",
    ):
        kf = KFold(n_splits=k, shuffle=True)
        results = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            self.to(device)
            self.fit(train_loader, criterion, optimizer, epochs, device)
            val_loss = self.evaluate(val_loader, criterion, device)
            results.append(val_loss)
            print(f"Fold {fold+1}/{k}, Validation Loss: {val_loss}")

        print(f"Mean Validation Loss: {sum(results) / k}")
        return results

    def evaluate(self, dataloader, loss, device="cpu"):
        self.to(device)
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = loss(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def tune(
        self,
        dataset,
        criterion,
        optimizer,
        param_grid,
        k=3,
        epochs=10,
        batch_size=32,
        device="cpu",
    ):
        raise NotImplementedError(
            "Not yet implemented: consider integration with e.g. ray tune"
        )


class SklearnBackend(BaseEstimator, BaseEmulator):
    def __init__(self, model):
        """Wraps an sklearn model while maintaining BaseModel's API."""
        self.model = model

    def fit(self, X, y, criterion=None, optimizer=None, epochs=10, device=None):
        """Trains the model. Criterion and optimizer are ignored (for API compatibility)."""
        self.model.fit(X, y)

    def predict(self, X, device=None):
        """Predicts output using the trained model."""
        return self.model.predict(X)

    def evaluate(self, X, y, criterion=None, device=None):
        """Evaluates the model's performance."""
        return self.model.score(X, y)

    def cross_validate(
        self,
        dataset,
        criterion=None,
        optimizer_class=None,
        k=5,
        epochs=10,
        batch_size=32,
        device=None,
    ):
        """Performs k-fold cross-validation."""
        X, y = dataset
        scores = cross_val_score(self.model, X, y, cv=k)
        mean_score = scores.mean()
        print(f"Cross-Validation Scores: {scores}, Mean: {mean_score}")
        return mean_score

    def tune(
        self,
        dataset,
        param_grid,
        criterion=None,
        optimizer_class=None,
        k=3,
        epochs=10,
        batch_size=32,
        device=None,
    ):
        """Hyperparameter tuning via GridSearch."""
        from sklearn.model_selection import GridSearchCV

        X, y = dataset
        grid_search = GridSearchCV(self.model, param_grid, cv=k)
        grid_search.fit(X, y)
        print(
            f"Best Params: {grid_search.best_params_}, Best Score: {grid_search.best_score_}"
        )
        return grid_search.best_params_


class InputTypeMixin:
    """A mixin for providing methods to handle type conversions"""

    def convert_to_tensor(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        return convert_to_tensor(X)

    def convert_to_numpy(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        return convert_to_numpy(X)
