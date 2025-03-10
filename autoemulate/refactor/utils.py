import numpy as np
from sklearn.datasets import make_regression
import torch


def convert_to_tensor(X: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(X, np.ndarray):
        return torch.tensor(X, dtype=torch.float32)
    return X


def convert_to_numpy(X: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        return X.numpy()
    return X


def sort_by_x(X, y):
    sorted_idx = np.argsort(X[:, 0])
    return X[sorted_idx], y[sorted_idx]


def sample_data_y1d():
    X, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=0)
    X, y = sort_by_x(X, y)
    return convert_to_tensor(X), convert_to_tensor(y)


def new_data_y1d():
    X, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=1)
    return convert_to_tensor(X), convert_to_tensor(y)


def sample_data_y2d():
    X, y = make_regression(n_samples=20, n_features=5, n_targets=2, random_state=0)
    return convert_to_tensor(X), convert_to_tensor(y)


def new_data_y2d():
    return make_regression(n_samples=20, n_features=5, n_targets=2, random_state=1)
