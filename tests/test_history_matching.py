from unittest.mock import patch

import numpy as np
import pytest
import torch

from autoemulate.experimental.types import TensorLike
from autoemulate.history_matching import HistoryMatching
from tests.test_base_simulator import MockSimulator

# Import the classes to test


@pytest.fixture
def mock_simulator():
    """Fixture for the mock simulator from test_base_simulator"""
    param_ranges = {"param1": (0.0, 1.0), "param2": (-10.0, 10.0)}
    return MockSimulator(param_ranges)


@pytest.fixture
def basic_observations():
    """Fixture for basic observation data matching mock simulator outputs"""
    return {"output1": (0.5, 0.1), "output2": (0.6, 0.2)}  # (mean, variance)


@pytest.fixture
def history_matcher(mock_simulator, basic_observations):
    """Fixture for a basic HistoryMatching instance using the mock simulator"""
    return HistoryMatching(
        simulator=mock_simulator,
        observations=basic_observations,
        threshold=3.0,
        model_discrepancy=0.1,
        rank=1,
    )


def test_predict_with_simulator(history_matcher, mock_simulator):
    """Test running a wave with the mock simulator"""
    parameter_samples = [
        {"param1": 0.1, "param2": 0.2},
        {"param1": 0.3, "param2": -0.4},
    ]

    X = torch.tensor(
        [
            [sample[name] for name in ["param1", "param2"]]
            for sample in parameter_samples
        ]
    )

    pred_means, pred_vars, successful_samples = history_matcher.predict(X)

    # With our mock simulator, all valid samples should succeed
    assert successful_samples.shape[0] == 2
    assert pred_means.shape[0] == 2
    assert successful_samples.shape == (2, 2)  # 2 samples, 2 outputs

    # When using simulator, no variance is returns
    assert pred_vars is None


# def test_predict_with_missing_params(history_matcher, mock_simulator):
#     """Test running a wave with invalid parameters that should fail"""
#     parameter_samples = [
#         {"param1": 0.1},  # Missing param2 - should fail
#         {"param1": 0.3, "param2": -0.4},  # Valid
#     ]

#     successful_samples, impl_scores = history_matcher.predict(
#         parameter_samples
#     )

#     # Only the valid sample should succeed
#     assert len(successful_samples) == 1
#     assert len(impl_scores) == 1
#     assert successful_samples[0] == parameter_samples[1]


def test_history_matcher_init(history_matcher, mock_simulator, basic_observations):
    """Test initialization of HistoryMatching with mock simulator"""
    assert history_matcher.simulator == mock_simulator
    # assert history_matcher.observations == basic_observations
    assert history_matcher.threshold == 3.0
    assert history_matcher.discrepancy == 0.1
    assert history_matcher.rank == 1


def test_calculate_implausibility(history_matcher):
    """Test implausibility calculation with mock simulator outputs"""

    # Shape [n_samples, n_outputs]
    pred_means = torch.Tensor([[0.4], [0.7]])
    pred_vars = torch.Tensor([[0.05], [0.1]])

    result = history_matcher.calculate_implausibility(pred_means, pred_vars)

    # Check the structure of the result
    assert set(result.keys()) == {"I", "NROY", "RO"}
    assert isinstance(result["I"], TensorLike)
    assert isinstance(result["NROY"], list)
    assert isinstance(result["RO"], list)
    assert len(result["I"]) == 2  # Should have implausibility for both outputs


@patch("tqdm.tqdm", lambda x, **kwargs: x)  # Mock tqdm to avoid progress bars in tests
def test_run(history_matcher):
    """Test the full history matching process with mock simulator"""
    n_waves = 2
    n_samples_per_wave = 5

    # Run history matching
    (
        all_samples,
        all_impl_scores,
        updated_emulator,
    ) = history_matcher.run(
        n_waves=n_waves, n_samples_per_wave=n_samples_per_wave, emulator_predict=False
    )

    # Check the basic structure of the results
    assert isinstance(all_samples, TensorLike)
    assert isinstance(all_impl_scores, TensorLike)
    assert updated_emulator is None  # Since we didn't use an emulator

    # We should get results for all valid samples
    assert len(all_samples) == n_waves * n_samples_per_wave
    assert len(all_impl_scores) == n_waves * n_samples_per_wave


def test_sample_nroy(history_matcher, mock_simulator):
    """Test generating new samples within NROY space using mock simulator"""

    nroy_samples = [
        {"param1": 0.1, "param2": 0.2},
        {"param1": 0.3, "param2": -0.4},
        {"param1": 0.2, "param2": 0.1},
    ]

    X_nroy = torch.Tensor(
        [[sample[name] for name in ["param1", "param2"]] for sample in nroy_samples]
    )

    n_samples = 5
    new_samples = history_matcher.sample_nroy(X_nroy, n_samples)

    # Check the number of samples
    assert new_samples.shape[0] == n_samples
    assert new_samples.shape[1] == len(mock_simulator.param_names)

    # Check that values are within the bounds of NROY samples
    param1_values = [s["param1"] for s in nroy_samples]
    param2_values = [s["param2"] for s in nroy_samples]

    assert (
        (min(param1_values) <= new_samples[:, 0])
        & (new_samples[:, 0] <= max(param1_values))
    ).all()

    assert (
        (min(param2_values) <= new_samples[:, 1])
        & (new_samples[:, 1] <= max(param2_values))
    ).all()
