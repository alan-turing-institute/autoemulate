import torch
from sklearn.model_selection import KFold, LeaveOneOut
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
            return torch.tensor(2 * [val * 2 for val in x])

        @staticmethod
        def get_tune_config():
            return {}

    x = torch.tensor([1, 2, 3, 4, 5])
    y = 2 * x
    dataset = TensorDataset(x, y)

    emulator = DummyEmulator()

    # KFold
    results = cross_validate(KFold(n_splits=2), dataset, emulator)
    assert "r2" in results
    assert "rmse" in results
    assert len(results["r2"]) == 2
    assert len(results["rmse"]) == 2

    # LOO
    results = cross_validate(LeaveOneOut(), dataset, emulator)
    assert len(results["r2"]) == 5
    assert len(results["rmse"]) == 5
