import numpy as np
import pytest

from autoemulate.history_matching import history_matching


@pytest.fixture
def sample_data_2d():
    pred_mean = np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1], [5.0, 5.1]])
    pred_std = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
    pred_var = np.square(pred_std)
    expectations = (pred_mean, pred_var)
    obs = [(1.5, 0.1), (2.5, 0.2)]
    return expectations, obs


@pytest.fixture
def sample_data_1d():
    pred_mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred_std = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    pred_var = np.square(pred_std)
    expectations = (pred_mean, pred_var)
    obs = [1.5, 10]
    return expectations, obs


def test_history_matching_1d(sample_data_1d):
    expectations, obs = sample_data_1d
    result = history_matching(expectations=expectations, obs=obs, threshold=1.0)
    assert "NROY" in result  # Ensure the key exists in the result
    assert isinstance(result["NROY"], list)  # Validate that NROY is a list
    assert len(result["NROY"]) > 0  # Ensure the list is not empty


def test_history_matching_threshold_1d(sample_data_1d):
    expectations, obs = sample_data_1d
    result = history_matching(expectations=expectations, obs=obs, threshold=0.5)
    assert "NROY" in result
    assert isinstance(result["NROY"], list)
    assert len(result["NROY"]) <= len(expectations[0])


def test_history_matching_2d(sample_data_2d):
    expectations, obs = sample_data_2d
    result = history_matching(expectations=expectations, obs=obs, threshold=1.0)
    assert "NROY" in result  # Ensure the key exists in the result
    assert isinstance(result["NROY"], list)  # Validate that NROY is a list
    assert len(result["NROY"]) > 0  # Ensure the list is not empty


def test_history_matching_threshold_2d(sample_data_2d):
    expectations, obs = sample_data_2d
    result = history_matching(expectations=expectations, obs=obs, threshold=0.5)
    assert "NROY" in result
    assert isinstance(result["NROY"], list)
    assert len(result["NROY"]) <= len(expectations[0])
