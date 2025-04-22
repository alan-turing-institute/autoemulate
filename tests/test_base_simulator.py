from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pytest

from autoemulate.simulations.base_simulator import BaseSimulator


# Mock implementation of BaseSimulator for testing
class MockSimulator(BaseSimulator):
    """Mock implementation of BaseSimulator for testing purposes"""

    def __init__(self, param_ranges):
        self._param_bounds = param_ranges
        self._param_names_list = list(param_ranges.keys())
        self._output_names_list = ["output1", "output2"]

    @property
    def param_names(self) -> List[str]:
        return self._param_names_list

    @property
    def output_names(self) -> List[str]:
        return self._output_names_list

    def generate_initial_samples(self, n_samples: int) -> List[Dict[str, float]]:
        """Generate mock samples"""
        samples = []
        for i in range(n_samples):
            sample = {}
            for name in self._param_names_list:
                min_val, max_val = self._param_bounds[name]
                sample[name] = min_val + (max_val - min_val) * np.random.random()
            samples.append(sample)
        return samples

    def run_simulation(self, params: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Run mock simulation"""
        # Check that all required parameters are present
        if not all(param in params for param in self._param_names_list):
            return None

        # Simple deterministic output based on input parameters
        try:
            param_sum = sum(params.values())
            return {"output1": param_sum / len(params), "output2": param_sum * 2}
        except Exception:
            return None


# Fixture to create a mock simulator with test parameters
@pytest.fixture
def mock_simulator():
    param_ranges = {"param1": (0.0, 1.0), "param2": (-10.0, 10.0), "param3": (0.5, 5.0)}
    return MockSimulator(param_ranges)


# Test creation of a concrete BaseSimulator implementation
def test_simulator_creation(mock_simulator):
    """Test that a concrete simulator can be created"""
    assert isinstance(mock_simulator, BaseSimulator)
    assert isinstance(mock_simulator, MockSimulator)


# Test param_names property returns expected values
def test_param_names(mock_simulator):
    """Test that param_names property returns correct parameter names"""
    expected = ["param1", "param2", "param3"]
    assert mock_simulator.param_names == expected
    assert len(mock_simulator.param_names) == 3


# Test output_names property returns expected values
def test_output_names(mock_simulator):
    """Test that output_names property returns correct output names"""
    expected = ["output1", "output2"]
    assert mock_simulator.output_names == expected
    assert len(mock_simulator.output_names) == 2


# Test generate_initial_samples method returns correct structure
def test_generate_initial_samples(mock_simulator):
    """Test that generate_initial_samples returns correct number and structure of samples"""
    n_samples = 5
    samples = mock_simulator.generate_initial_samples(n_samples)

    # Check number of samples
    assert len(samples) == n_samples

    # Check structure of each sample
    for sample in samples:
        assert isinstance(sample, dict)
        assert set(sample.keys()) == set(mock_simulator.param_names)

        # Check parameter bounds
        assert 0.0 <= sample["param1"] <= 1.0
        assert -10.0 <= sample["param2"] <= 10.0
        assert 0.5 <= sample["param3"] <= 5.0


# Test run_simulation method returns correct output structure
def test_run_simulation(mock_simulator):
    """Test that run_simulation returns correct output structure"""
    # Create a test parameter set
    params = {"param1": 0.5, "param2": 0.0, "param3": 2.5}

    # Run simulation
    result = mock_simulator.run_simulation(params)

    # Check result structure
    assert isinstance(result, dict)
    assert set(result.keys()) == set(mock_simulator.output_names)

    # Check specific values for this deterministic mock
    expected_output1 = (0.5 + 0.0 + 2.5) / 3
    expected_output2 = (0.5 + 0.0 + 2.5) * 2
    assert result["output1"] == pytest.approx(expected_output1)
    assert result["output2"] == pytest.approx(expected_output2)


# Test run_simulation with invalid parameters
def test_run_simulation_invalid_params(mock_simulator):
    """Test that run_simulation handles invalid parameters gracefully"""
    # Create an incomplete parameter set
    params = {
        "param1": 0.5,
        # Missing param2 and param3
    }

    # Run simulation with invalid parameters
    result = mock_simulator.run_simulation(params)

    # We expect it to return None for invalid parameters
    assert result is None


# Test attempting to instantiate abstract base class
def test_abstract_class_instantiation():
    """Test that BaseSimulator cannot be instantiated directly"""
    with pytest.raises(TypeError):
        BaseSimulator()  # Should raise TypeError


# Integration test - generate samples and run simulations on them
def test_end_to_end_workflow(mock_simulator):
    """Test the end-to-end workflow of generating samples and running simulations"""
    # Generate samples
    n_samples = 10
    samples = mock_simulator.generate_initial_samples(n_samples)

    # Run simulations on all samples
    results = []
    for sample in samples:
        result = mock_simulator.run_simulation(sample)
        if result is not None:
            results.append(result)

    # All simulations should succeed with our mock
    assert len(results) == n_samples

    # Each result should have the expected structure
    for result in results:
        assert isinstance(result, dict)
        assert set(result.keys()) == set(mock_simulator.output_names)
        assert "output1" in result
        assert "output2" in result
