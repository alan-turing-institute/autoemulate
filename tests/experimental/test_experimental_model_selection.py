import torch
from sklearn.model_selection import KFold, LeaveOneOut
from torch.utils.data import TensorDataset

from autoemulate.experimental.model_selection import cross_validate


def test_cross_validate():
    """
    Test cross_validate can be called with any sklearn.model_selection class.
    """

    class SimpleEmulator:
        def fit(self, train_loader):
            pass

        def predict(self, X):
            return torch.tensor([x * 2 for x in X])

    x = torch.tensor([1, 2, 3, 4, 5])
    y = 2 * x

    # KFold
    results = cross_validate(KFold(n_splits=2), TensorDataset(x, y), SimpleEmulator())
    assert "r2" in results
    assert "rmse" in results
    assert len(results["r2"]) == 2
    assert len(results["rmse"]) == 2

    # LOO
    results = cross_validate(LeaveOneOut(), TensorDataset(x, y), SimpleEmulator())
    assert len(results["r2"]) == 5
    assert len(results["rmse"]) == 5
