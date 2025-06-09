from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pytest

from autoemulate.simulations.base import Simulator
from autoemulate.simulations.epidemic import EpidemicSimulator
from autoemulate.simulations.epidemic import simulate_epidemic
from autoemulate.simulations.projectile import ProjectileSimulator
from autoemulate.simulations.projectile import simulate_projectile

# In test_base_simulator.py, update the MockSimulator class:


class MockSimulator(Simulator):
    """Mock implementation of Simulator for testing purposes"""

    def __init__(self, parameters_range, output_variables=None):
        # Properly call the parent class constructor
        super().__init__(parameters_range, output_variables)

        # If output_variables not provided, set default ones
        if not self._output_variables:
            self._output_variables = ["output1", "output2"]

        # Initialize output_names based on base class expectations
        self._output_names = self._output_variables.copy()
        self._has_sample_forward = True

    def sample_forward(self, params: Dict[str, float]) -> Optional[np.ndarray]:
        """Run mock simulation and return numpy array as specified in the base class"""
        # Check that all required parameters are present
        if not all(param in params for param in self._param_names):
            return None

        # Simple deterministic output based on input parameters
        try:
            param_sum = sum(params.values())
            # Return as numpy array as required by base class
            return np.array(
                [param_sum / len(params), param_sum * 2]  # output1  # output2
            )
        except Exception:
            return None


# Fixture to create a mock simulator with test parameters
@pytest.fixture
def mock_simulator():
    param_ranges = {"param1": (0.0, 1.0), "param2": (-10.0, 10.0), "param3": (0.5, 5.0)}
    return MockSimulator(param_ranges)


# Test creation of a concrete Simulator implementation
def test_simulator_creation(mock_simulator):
    """Test that a concrete simulator can be created"""
    assert isinstance(mock_simulator, Simulator)
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


# Test sample_inputs method returns correct structure
def test_sample_inputs(mock_simulator):
    """Test that sample_inputs returns correct number and structure of samples"""
    n_samples = 5
    samples = mock_simulator.sample_inputs(n_samples)

    # Check number of samples
    assert len(samples) == n_samples

    # Check structure of each sample
    for sample in samples:
        assert isinstance(sample, np.ndarray)

        # Check parameter bounds
        assert 0.0 <= sample[0] <= 1.0
        assert -10.0 <= sample[1] <= 10.0
        assert 0.5 <= sample[2] <= 5.0


# Test sample_forward method returns correct output structure
def test_sample_forward(mock_simulator):
    """Test that sample_forward returns correct output structure"""
    # Create a test parameter set
    params = {"param1": 0.5, "param2": 0.0, "param3": 2.5}

    # Run simulation
    result = mock_simulator.sample_forward(params)

    # Check result structure - should be numpy array as specified in base class
    assert isinstance(result, np.ndarray)
    assert len(result) == len(mock_simulator.output_names)

    # Check specific values for this deterministic mock
    expected_output1 = (0.5 + 0.0 + 2.5) / 3
    expected_output2 = (0.5 + 0.0 + 2.5) * 2
    assert result[0] == pytest.approx(expected_output1)
    assert result[1] == pytest.approx(expected_output2)


# Test sample_forward with invalid parameters
def test_sample_forward_invalid_params(mock_simulator):
    """Test that sample_forward handles invalid parameters gracefully"""
    # Create an incomplete parameter set
    params = {
        "param1": 0.5,
        # Missing param2 and param3
    }

    # Run simulation with invalid parameters
    result = mock_simulator.sample_forward(params)

    # We expect it to return None for invalid parameters
    assert result is None


# Test attempting to instantiate abstract base class
def test_abstract_class_instantiation():
    """Test that Simulator cannot be instantiated directly"""
    with pytest.raises(TypeError):
        Simulator()  # Should raise TypeError


# Integration test - generate samples and run simulations on them
def test_end_to_end_workflow(mock_simulator):
    """Test the end-to-end workflow of generating samples and running simulations"""
    # Generate samples
    n_samples = 10
    samples = mock_simulator.sample_inputs(n_samples)
    samples = mock_simulator.convert_samples(samples)

    # Run simulations on all samples
    results = []
    for sample in samples:
        result = mock_simulator.sample_forward(sample)
        if result is not None:
            results.append(result)

    # All simulations should succeed with our mock
    assert len(results) == n_samples

    # Each result should have the expected structure
    for result in results:
        assert isinstance(result, np.ndarray)
        assert len(result) == len(mock_simulator.output_names)

    # Test the get_results_dataframe method
    results_array = np.array(results)
    df = mock_simulator.get_results_dataframe(samples, results_array)

    # Check DataFrame structure
    assert len(df) == n_samples

    # Should have columns for both parameters and outputs
    assert set(mock_simulator.param_names).issubset(set(df.columns))
    assert set(mock_simulator.output_names).issubset(set(df.columns)) or set(
        f"output_{i}" for i in range(len(mock_simulator.output_names))
    ).issubset(set(df.columns))


def test_projectile_simulator():
    """
    Sense check ProjectileSimulator against previous implementation.
    """
    sim = ProjectileSimulator()
    X = sim.sample_inputs(100)
    y = sim.run_batch_simulations(X)
    y_old = np.array([simulate_projectile(x) for x in X])
    assert np.allclose(y, y_old)


def test_epidemoc_simulator():
    """
    Sense check EpidemicSimulator against previous implementation.
    """
    sim = EpidemicSimulator()
    X = sim.sample_inputs(100)
    y = sim.run_batch_simulations(X)
    y_old = np.array([simulate_epidemic(x) for x in X])
    assert np.allclose(y, y_old)
