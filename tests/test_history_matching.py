from unittest.mock import patch

import numpy as np
import pytest

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


def test_run_wave_with_simulator(history_matcher, mock_simulator):
    """Test running a wave with the mock simulator"""
    parameter_samples = [
        {"param1": 0.1, "param2": 0.2},
        {"param1": 0.3, "param2": -0.4},
    ]

    successful_samples, impl_scores = history_matcher.run_wave(
        parameter_samples, use_emulator=False
    )

    # With our mock simulator, all valid samples should succeed
    assert len(successful_samples) == 2
    assert len(impl_scores) == 2

    # Check the implausibility scores shape
    assert impl_scores.shape == (2, 2)  # 2 samples, 2 outputs


# def test_run_wave_with_missing_params(history_matcher, mock_simulator):
#     """Test running a wave with invalid parameters that should fail"""
#     parameter_samples = [
#         {"param1": 0.1},  # Missing param2 - should fail
#         {"param1": 0.3, "param2": -0.4},  # Valid
#     ]

#     successful_samples, impl_scores = history_matcher.run_wave(
#         parameter_samples, use_emulator=False
#     )

#     # Only the valid sample should succeed
#     assert len(successful_samples) == 1
#     assert len(impl_scores) == 1
#     assert successful_samples[0] == parameter_samples[1]


def test_history_matcher_init(history_matcher, mock_simulator, basic_observations):
    """Test initialization of HistoryMatching with mock simulator"""
    assert history_matcher.simulator == mock_simulator
    assert history_matcher.observations == basic_observations
    assert history_matcher.threshold == 3.0
    assert history_matcher.discrepancy == 0.1
    assert history_matcher.rank == 1


def test_calculate_implausibility(history_matcher):
    """Test implausibility calculation with mock simulator outputs"""

    # Shape [n_samples, n_outputs]
    pred_means = np.array([[0.4], [0.7]])
    pred_vars = np.array([[0.05], [0.1]])

    result = history_matcher.calculate_implausibility(pred_means, pred_vars)

    # Check the structure of the result
    assert set(result.keys()) == {"I", "NROY", "RO"}
    assert isinstance(result["I"], np.ndarray)
    assert isinstance(result["NROY"], list)
    assert isinstance(result["RO"], list)
    assert len(result["I"]) == 2  # Should have implausibility for both outputs


def test_run_wave_with_simulator(history_matcher, mock_simulator):
    """Test running a wave with the mock simulator"""
    parameter_samples = [
        {"param1": 0.1, "param2": 0.2},
        {"param1": 0.3, "param2": -0.4},
    ]

    successful_samples, impl_scores = history_matcher.run_wave(
        parameter_samples, use_emulator=False
    )

    # With our mock simulator, all samples should succeed
    assert len(successful_samples) == 2
    assert len(impl_scores) == 2

    # Check the implausibility scores shape
    assert impl_scores.shape == (2, 2)  # 2 samples, 2 outputs


@patch("tqdm.tqdm", lambda x, **kwargs: x)  # Mock tqdm to avoid progress bars in tests
def test_run_history_matching(history_matcher):
    """Test the full history matching process with mock simulator"""
    n_waves = 2
    n_samples_per_wave = 5

    # Run history matching
    (
        all_samples,
        all_impl_scores,
        updated_emulator,
    ) = history_matcher.run_history_matching(
        n_waves=n_waves, n_samples_per_wave=n_samples_per_wave, use_emulator=False
    )

    # Check the basic structure of the results
    assert isinstance(all_samples, list)
    assert isinstance(all_impl_scores, np.ndarray)
    assert updated_emulator is None  # Since we didn't use an emulator

    # We should get results for all valid samples
    assert len(all_samples) == n_waves * n_samples_per_wave
    assert len(all_impl_scores) == n_waves * n_samples_per_wave


def test_generate_new_samples(history_matcher, mock_simulator):
    """Test generating new samples within NROY space using mock simulator"""
    nroy_samples = [
        {"param1": 0.1, "param2": 0.2},
        {"param1": 0.3, "param2": -0.4},
        {"param1": 0.2, "param2": 0.1},
    ]

    n_samples = 5
    new_samples = history_matcher.generate_new_samples(nroy_samples, n_samples)

    # Check the number of samples
    assert len(new_samples) == n_samples

    # Check that all samples are dictionaries with the right keys
    for sample in new_samples:
        assert isinstance(sample, dict)
        assert set(sample.keys()) == set(mock_simulator.param_names)

    # Check that values are within the bounds of NROY samples
    param1_values = [s["param1"] for s in nroy_samples]
    param2_values = [s["param2"] for s in nroy_samples]

    for sample in new_samples:
        assert min(param1_values) <= sample["param1"] <= max(param1_values)
        assert min(param2_values) <= sample["param2"] <= max(param2_values)
