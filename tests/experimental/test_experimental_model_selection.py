import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset

from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.model_selection import cross_validate


def test_cross_validate():
    """
    Test cross_validate can be called with any sklearn.model_selection class.
    """

    class DummyEmulator(Emulator):
        def fit(self, x, y):
            pass

        def predict(self, x):
            return torch.tensor([val * 2 for val in x])

        @staticmethod
        def get_tune_config():
            return {}

    x = torch.tensor(np.arange(128)).float()
    y = 2 * x
    dataset = TensorDataset(x, y)

    emulator = DummyEmulator()

    # KFold
    results = cross_validate(KFold(n_splits=2), dataset, emulator)
    assert "r2" in results
    assert "rmse" in results
    assert len(results["r2"]) == 2
    assert len(results["rmse"]) == 2

    # LOO raised an error with torchmetrics R2Score since it requires at least 2 samples
    # KFold with more splits
    results = cross_validate(KFold(n_splits=x.shape[0] // 2), dataset, emulator)
    assert len(results["r2"]) == x.shape[0] // 2
    assert len(results["rmse"]) == x.shape[0] // 2
