import numpy as np
import torch
from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.model_selection import cross_validate
from autoemulate.emulators.base import Emulator
from sklearn.model_selection import KFold, LeavePOut
from torch import nn
from torch.utils.data import TensorDataset


def test_cross_validate():
    """
    Test cross_validate can be called with any sklearn.model_selection class.
    """

    class DummyEmulator(Emulator, torch.nn.Module):
        def __init__(self, x=None, y=None, device=None, **kwargs):
            nn.Module.__init__(self)
            TorchDeviceMixin.__init__(self, device=device)
            _, _ = x, y

        def _fit(self, x, y, validation_data=None):
            pass

        def _predict(self, x, with_grad=False):
            """
            Dummy predict that always returns
            a 2D tensor.
            """
            if with_grad:
                msg = "Error!"
                raise ValueError(msg)
            return x

        @staticmethod
        def get_tune_params():
            return {}

        @staticmethod
        def is_multioutput():
            return False

    # Make 2D
    x = torch.tensor(np.arange(32)).unsqueeze(1).float()
    y = x  # Dummy target, same shape as _predict output
    dataset = TensorDataset(x, y)

    emulator_cls = DummyEmulator

    # KFold
    results = cross_validate(KFold(n_splits=2), dataset, emulator_cls, {})
    assert "r2" in results
    assert "rmse" in results
    assert len(results["r2"]) == 2
    assert len(results["rmse"]) == 2

    # LeavePOut: LOO raised an error with torchmetrics R2Score since it requires at
    # least 2 samples
    results = cross_validate(LeavePOut(p=2), dataset, emulator_cls, {})
    expected_n = (x.shape[0] * (x.shape[0] - 1)) / 2
    assert len(results["r2"]) == expected_n
    assert len(results["rmse"]) == expected_n
