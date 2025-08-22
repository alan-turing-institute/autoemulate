import pytest
import torch
from autoemulate.core.types import TensorLike
from autoemulate.simulations.base import Simulator
from torch import Tensor


class MockSimulator(Simulator):
    """Mock implementation of Simulator for testing purposes"""

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]],
        output_names: list[str],
    ):
        # Call parent constructor
        super().__init__(parameters_range, output_names)

    def _forward(self, x: TensorLike) -> TensorLike | None:
        """
        Implement abstract _forward method with a simple transformation.
        Input shape: (n_samples, n_features)
        Output shape: (n_samples, n_outputs)
        """
        # Create different outputs for each variable
        outputs = []
        for i, _ in enumerate(self._output_names):
            # Create a unique output for each variable
            output = torch.sum(x, dim=1) * (i + 1)
            outputs.append(output.view(-1, 1))

        # Concatenate all outputs along dimension 1
        return torch.cat(outputs, dim=1)


@pytest.fixture
def parameters_range():
    """Create test parameter ranges"""
    return {"param1": (0.0, 1.0), "param2": (-1.0, 1.0), "param3": (5.0, 10.0)}


@pytest.fixture
def output_names():
    """Create test output names"""
    return ["var1", "var2"]


@pytest.fixture
def mock_simulator(parameters_range, output_names):
    """Create a mock simulator instance"""
    return MockSimulator(parameters_range, output_names)


def test_simulator_init(mock_simulator, parameters_range, output_names):
    """Test that the simulator initializes correctly with provided values"""
    # Check param names and bounds
    assert mock_simulator._param_names == list(parameters_range.keys())
    assert mock_simulator._param_bounds == list(parameters_range.values())
    assert mock_simulator.in_dim == 3

    # Check output_names
    assert mock_simulator._output_names == output_names
    assert mock_simulator.out_dim == 2

    # Check output names are generated correctly in mock implementation
    assert mock_simulator._output_names == ["var1", "var2"]

    # Check has_sample_forward flag
    assert mock_simulator._has_sample_forward is False


def test_sample_inputs(mock_simulator):
    """Test that sample_inputs generates correct samples"""
    n_samples = 5
    samples = mock_simulator.sample_inputs(n_samples)

    # Check return type
    assert isinstance(samples, Tensor)

    # Check shape
    assert samples.shape == (n_samples, len(mock_simulator._param_names))

    # Check values are within bounds
    for i, param_name in enumerate(mock_simulator._param_names):
        min_val, max_val = mock_simulator._parameters_range[param_name]
        assert torch.all(samples[:, i] >= min_val)
        assert torch.all(samples[:, i] <= max_val)


def test_forward(mock_simulator):
    """Test that forward method works correctly"""
    # Create test input
    test_input = torch.tensor([[0.5, 0.0, 7.5]], dtype=torch.float32)

    # Get output
    output = mock_simulator.forward(test_input)

    # Check shape
    assert output.shape == (1, len(mock_simulator._output_names))

    # Check values against expected transformation
    expected_sum = 0.5 + 0.0 + 7.5
    assert output[0, 0] == pytest.approx(expected_sum * 1)
    assert output[0, 1] == pytest.approx(expected_sum * 2)


def test_forward_batch(mock_simulator):
    """Test that forward_batch method processes multiple samples correctly"""
    # Create test batch
    n_samples = 3
    batch = torch.tensor(
        [[0.5, 0.0, 7.5], [0.2, 0.3, 6.0], [0.8, -0.5, 9.0]], dtype=torch.float32
    )

    # Process batch
    results = mock_simulator.forward_batch(batch)

    # Check shape
    assert results.shape == (n_samples, len(mock_simulator._output_names))

    # Check values for each sample
    for i in range(n_samples):
        expected_sum = sum(batch[i].tolist())
        assert results[i, 0] == pytest.approx(expected_sum * 1)
        assert results[i, 1] == pytest.approx(expected_sum * 2)


def test_get_parameter_idx(mock_simulator):
    """Test that get_parameter_idx returns correct indices"""
    assert mock_simulator.get_parameter_idx("param1") == 0
    assert mock_simulator.get_parameter_idx("param2") == 1
    assert mock_simulator.get_parameter_idx("param3") == 2

    # Test invalid parameter
    with pytest.raises(ValueError, match="Parameter .* not found"):
        mock_simulator.get_parameter_idx("invalid_param")


def test_properties(mock_simulator, parameters_range, output_names):
    """Test that properties return correct values"""
    # Test param_names
    assert mock_simulator.param_names == list(parameters_range.keys())

    # Test param_bounds
    assert mock_simulator.param_bounds == list(parameters_range.values())

    # Test output_names
    assert mock_simulator.output_names == ["var1", "var2"]

    # Test output_names
    assert mock_simulator.output_names == output_names


def test_abstract_class():
    """Test that Simulator cannot be instantiated directly"""
    params = {"param": (0, 1)}
    outputs = ["out"]

    # Should raise TypeError when trying to instantiate abstract base class
    with pytest.raises(TypeError):
        Simulator(params, outputs)  # type: ignore[abstract]


def test_handle_simulation_failure():
    """Test handling of simulation failures in forward_batch"""

    class ThresholdSimulator(MockSimulator):
        def _forward(self, x: TensorLike) -> TensorLike | None:
            # Only process inputs where the first value is > 0.5
            if x[0, 0] > 0.5:
                return super()._forward(x)
            return None

    # Create simulator with float parameters
    params = {"param1": (0.0, 1.0), "param2": (0.0, 1.0), "param3": (0.0, 1.0)}
    simulator = ThresholdSimulator(params, ["var1"])

    # Create mixed batch with varying first values
    batch = torch.tensor(
        [
            [0.2, 0.5, 0.5],  # Below threshold
            [0.6, 0.5, 0.5],  # Above threshold
            [0.1, 0.5, 0.5],  # Below threshold
            [0.7, 0.5, 0.5],  # Above threshold
        ],
        dtype=torch.float32,
    )

    # This should process all samples without errors
    # We're just verifying it doesn't crash
    results, valid_x = simulator.forward_batch_skip_failures(batch)
    assert isinstance(results, TensorLike)

    # Verify results shape
    assert results.shape == (2, 1)
    assert valid_x.shape == (2, 3)


def test_update_parameters_range(mock_simulator):
    """Test that parameters_range can be updated"""
    new_range = {"param1": (0.1, 0.9), "param2": (-0.5, 0.5), "param3": (4.0, 6.0)}
    mock_simulator.parameters_range = new_range
    assert mock_simulator.parameters_range == new_range
    assert mock_simulator.param_bounds == list(new_range.values())
    assert mock_simulator.in_dim == len(new_range)


def test_update_output_names(mock_simulator):
    """Test that output_names can be updated with same number of outputs"""
    new_output_names = ["var1_new", "var2_new"]
    mock_simulator.output_names = new_output_names
    assert mock_simulator.output_names == new_output_names
    assert mock_simulator.out_dim == 2


def test_update_output_names_wrong_dimension(mock_simulator):
    """Test that setting output_names with wrong dimension raises error"""
    with pytest.raises(ValueError, match="Number of output names \\(1\\) must match"):
        mock_simulator.output_names = ["only_one_var"]

    with pytest.raises(ValueError, match="Number of output names \\(3\\) must match"):
        mock_simulator.output_names = ["var1", "var2", "var3"]

    # Verify original names are unchanged after failed attempts
    assert mock_simulator.output_names == ["var1", "var2"]
    assert mock_simulator.out_dim == 2


def test_sample():
    param_bouns = {
        "param1": (0.0, 1.0),
        "param2": (10.0, 100.0),
        "param3": (1000.0, 1000.0),
    }
    sim = MockSimulator(param_bouns, ["var1", "var2"], method)

    n_samples = 1000
    samples = sim.sample_inputs(n_samples)

    assert isinstance(samples, TensorLike)
    assert samples.shape[0] == n_samples
    assert samples.shape[1] == 3
    assert torch.all(samples[:, 0] >= 0.0)
    assert torch.all(samples[:, 0] <= 1.0)
    assert torch.all(samples[:, 1] >= 10.0)
    assert torch.all(samples[:, 1] <= 100.0)
    assert torch.all(samples[:, 1] <= 100.0)
    assert torch.all(samples[:, 2] == 1000.0)
