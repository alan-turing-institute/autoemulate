from unittest.mock import MagicMock

import pytest
import torch
from autoemulate.callbacks.early_stopping import EarlyStopping
from autoemulate.emulators.gaussian_process.exact import (
    GaussianProcess,
)
from gpytorch.likelihoods import MultitaskGaussianLikelihood


@pytest.fixture
def gp_exact():
    x = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return GaussianProcess(
        x=x,
        y=y,
        likelihood_cls=MultitaskGaussianLikelihood,
        epochs=5,
        batch_size=2,
        lr=0.1,
        early_stopping=None,
    )


def test_early_stopping_method_call_counts(gp_exact):
    """
    Test that early stopping methods are called the correct number of times.
    """
    early_stopping = MagicMock(spec=EarlyStopping)
    gp_exact.early_stopping = early_stopping

    x = torch.randn(10, 3)
    y = torch.randn(10, 2)
    gp_exact._fit(x, y)

    early_stopping.on_train_begin.assert_called_once()
    early_stopping.on_train_end.assert_called_once()
    assert early_stopping.on_epoch_end.call_count == gp_exact.epochs


def test_early_stopping_trigger(gp_exact):
    """Test that early stopping is triggered."""
    # Set very high threshold and low patience to ensure it gets triggered
    early_stopping = EarlyStopping(threshold=100, patience=2)
    gp_exact.early_stopping = early_stopping

    x = torch.randn(10, 3)
    y = torch.randn(10, 2)
    gp_exact._fit(x, y)

    assert early_stopping.misses_ == early_stopping.patience
