from unittest.mock import patch

import pytest
import torch
from autoemulate.experimental.history_matching import HistoryMatching
from autoemulate.experimental.types import TensorLike

from .test_experimental_base_simulator import MockSimulator

# Import the classes to test


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
def history_matcher(mock_simulator, observations):
    """Fixture for a basic HistoryMatching instance using the mock simulator"""
    return HistoryMatching(
        simulator=mock_simulator,
        observations=observations,
        threshold=3.0,
        model_discrepancy=0.1,
        rank=1,
    )


def test_predict_with_simulator(history_matcher):
    """Test running a wave with the mock simulator"""

    x = torch.tensor([[0.1, 0.2], [0.3, -0.4]])  # [n_sample, n_output]
    pred_means, pred_vars, successful_samples = history_matcher.predict(x)

    # With our mock simulator, all valid samples should succeed
    assert successful_samples.shape[0] == 2
    assert pred_means.shape[0] == 2
    assert successful_samples.shape == (2, 2)  # 2 samples, 2 outputs

    # When using simulator, no variance is returns
    assert pred_vars is None


def test_history_matcher_init(history_matcher, mock_simulator):
    """Test initialization of HistoryMatching with mock simulator"""
    assert history_matcher.simulator == mock_simulator
    assert history_matcher.threshold == 3.0
    assert history_matcher.discrepancy == 0.1
    assert history_matcher.rank == 1
    assert history_matcher.obs_means.shape == (1, 2)
    assert history_matcher.obs_vars.shape == (1, 2)


def test_calculate_implausibility(history_matcher, observations):
    """Test implausibility calculation with mock simulator outputs"""

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


def test_invalid_inputs(history_matcher):
    # TODO
    pass


@patch("tqdm.tqdm", lambda x, **kwargs: x)  # Mock tqdm to avoid progress bars in tests
def test_run(history_matcher):
    """Test the full history matching process with a mock simulator"""
    n_waves = 2
    n_samples_per_wave = 5

    # Run history matching
    updated_emulator = history_matcher.run(
        n_waves=n_waves, n_samples_per_wave=n_samples_per_wave, emulator_predict=False
    )
    all_samples = history_matcher.tested_params
    all_impl_scores = history_matcher.impl_scores

    # Check basic structure of results
    assert isinstance(all_samples, TensorLike)
    assert isinstance(all_impl_scores, TensorLike)
    assert updated_emulator is None  # Since we didn't use an emulator

    # We should get results for all valid samples
    assert len(all_samples) == n_waves * n_samples_per_wave
    assert len(all_impl_scores) == n_waves * n_samples_per_wave


def test_sample_nroy(history_matcher, mock_simulator):
    """Test generating new samples within NROY space using mock simulator"""

    X_nroy = torch.tensor([[0.1, 0.2], [0.3, -0.4], [0.2, 0.1]])

    n_samples = 5
    new_samples = history_matcher.sample_nroy(n_samples, X_nroy)

    # Check the number of samples
    assert new_samples.shape[0] == n_samples
    assert new_samples.shape[1] == len(mock_simulator.param_names)

    # Check that values are within the bounds of NROY samples
    assert (
        (torch.min(X_nroy[:, 0]) <= new_samples[:, 0])
        & (new_samples[:, 0] <= torch.max(X_nroy[:, 0]))
    ).all()

    assert (
        (torch.min(X_nroy[:, 1]) <= new_samples[:, 1])
        & (new_samples[:, 1] <= torch.max(X_nroy[:, 1]))
    ).all()
