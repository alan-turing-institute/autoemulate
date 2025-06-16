from unittest.mock import patch

import pytest
import torch
from autoemulate.experimental.emulators.gaussian_process.exact import (
    GaussianProcessExact,
)
from autoemulate.experimental.history_matching import (
    HistoryMatching,
    HistoryMatchingWorkflow,
)
from autoemulate.experimental.types import TensorLike

from .test_experimental_base_simulator import MockSimulator


@pytest.fixture
def mock_simulator():
    """Fixture for the mock simulator from test_base_simulator"""
    param_ranges = {"param1": (0.0, 1.0), "param2": (-10.0, 10.0)}
    return MockSimulator(param_ranges, output_names=["output1", "output2"])


@pytest.fixture
def observations():
    """Fixture for observations data matching mock simulator outputs"""
    return {"output1": (0.5, 0.1), "output2": (0.6, 0.2)}  # (mean, variance)


@pytest.fixture
def history_matcher(observations):
    """Fixture for a basic HistoryMatching instance."""
    return HistoryMatching(
        observations=observations,
        threshold=3.0,
        model_discrepancy=0.1,
        rank=1,
    )


def test_history_matcher_init(history_matcher):
    """Test initialization of HistoryMatching."""
    assert history_matcher.threshold == 3.0
    assert history_matcher.discrepancy == 0.1
    assert history_matcher.rank == 1
    assert history_matcher.obs_means.shape == (1, 2)
    assert history_matcher.obs_vars.shape == (1, 2)


def test_calculate_implausibility(history_matcher, observations):
    """Test implausibility calculation."""

    # have 1 sample of 2 outputs arranged as [n_samples, n_outputs]
    pred_means = torch.Tensor([[0.4, 0.7]])
    pred_vars = torch.Tensor([[0.05, 0.1]])

    impl_scores = history_matcher.calculate_implausibility(pred_means, pred_vars)
    assert impl_scores.shape == (1, 2)

    assert impl_scores[0][0] == (
        abs(pred_means[0][0] - observations["output1"][0])
        # have an extra term in the denominator for model discrepancy
        / (pred_vars[0][0] + observations["output1"][1] + 0.1) ** 0.5
    )

    assert impl_scores[0][1] == (
        abs(pred_means[0][1] - observations["output2"][0])
        # have an extra term in the denominator for model discrepancy
        / (pred_vars[0][1] + observations["output2"][1] + 0.1) ** 0.5
    )


def test_get_indices(history_matcher):
    """Test NROY and RO indices vary with rank."""
    impl_scores = torch.tensor([[1, 5], [1, 2], [4, 2]])

    # rank = 1
    nroy = history_matcher.get_nroy(impl_scores)
    assert len(nroy) == 1
    assert nroy[0] == 1

    ro = history_matcher.get_ro(impl_scores)
    assert len(ro) == 2
    assert ro[0] == 0
    assert ro[1] == 2

    # rank = n
    history_matcher.rank = 2
    assert len(history_matcher.get_nroy(impl_scores)) == 3
    assert len(history_matcher.get_ro(impl_scores)) == 0


def test_sample_nroy(history_matcher):
    """Test generating new samples within NROY space using mock simulator"""

    X_nroy = torch.tensor([[0.1, 0.2], [0.3, -0.4], [0.2, 0.1]])

    n_samples = 5
    new_samples = history_matcher.sample_nroy(n_samples, X_nroy)

    # Check the number of samples
    assert new_samples.shape[0] == n_samples
    assert new_samples.shape[1] == history_matcher.out_dim

    # Check that values are within the bounds of NROY samples
    assert (
        (torch.min(X_nroy[:, 0]) <= new_samples[:, 0])
        & (new_samples[:, 0] <= torch.max(X_nroy[:, 0]))
    ).all()

    assert (
        (torch.min(X_nroy[:, 1]) <= new_samples[:, 1])
        & (new_samples[:, 1] <= torch.max(X_nroy[:, 1]))
    ).all()


@patch("tqdm.tqdm", lambda x, **kwargs: x)  # Mock tqdm to avoid progress bars in tests
def test_run(observations, mock_simulator):
    """Test the full history matching workflow with a mock simulator"""
    x = torch.tensor([[0.1, 0.2], [0.3, -0.4]])
    y = mock_simulator.forward_batch(x)

    # Run history matching
    gp = GaussianProcessExact(x, y)
    gp.fit(x, y)

    hm = HistoryMatchingWorkflow(
        simulator=mock_simulator,
        emulator=gp,
        observations=observations,
        threshold=3.0,
        model_discrepancy=0.1,
        rank=1,
    )

    # call run first time
    hm.run(n_samples=5)

    # Check basic structure of results
    assert isinstance(hm.tested_params, TensorLike)
    assert isinstance(hm.impl_scores, TensorLike)
    assert hm.emulator is not None

    assert len(hm.tested_params) == 5
    assert len(hm.impl_scores) == 5

    # can run again
    hm.run(n_samples=5)

    # We should get results for all valid samples
    assert len(hm.tested_params) == 5 * 2
    assert len(hm.impl_scores) == 5 * 2
