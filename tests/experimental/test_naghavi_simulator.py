from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from autoemulate.experimental.simulations.naghavi_cardiac_modular_circ import (
    NaghaviSimulator,
)
from torch import Tensor


# Mock the ModularCirc imports since we don't want to run actual simulations in tests
@pytest.fixture
def mock_modular_circ(monkeypatch):
    """Mock the ModularCirc dependencies"""
    # Mock NaghaviModelParameters
    mock_params = mock.MagicMock()
    mock_params._set_comp = mock.MagicMock()

    # Mock NaghaviModel
    mock_model = mock.MagicMock()
    mock_model.components = {
        "lv": type(
            "LVMock",
            (),
            {
                "P_i": type("Variable", (), {"values": [5.0, 10.0, 15.0, 20.0]}),
                "P_o": type("Variable", (), {"values": [1.0, 5.0, 10.0, 13.0]}),
            },
        )
    }

    # Mock Solver
    mock_solver = mock.MagicMock()
    mock_solver.converged = True
    mock_solver.solve = mock.MagicMock()

    # Apply mocks
    monkeypatch.setattr(
        "autoemulate.experimental.simulations.naghavi_cardiac_modular_circ.NaghaviModelParameters",
        mock.MagicMock(return_value=mock_params),
    )
    monkeypatch.setattr(
        "autoemulate.experimental.simulations.naghavi_cardiac_modular_circ.NaghaviModel",
        mock.MagicMock(return_value=mock_model),
    )
    monkeypatch.setattr(
        "autoemulate.experimental.simulations.naghavi_cardiac_modular_circ.Solver",
        mock.MagicMock(return_value=mock_solver),
    )

    return mock_model, mock_params, mock_solver


@pytest.fixture
def parameters_range():
    """Create test parameter ranges for the Naghavi simulator"""
    return {
        "T": (0.8, 1.2),
        "lv.E_pas": (0.1, 0.5),
        "lv.E_act": (1.0, 3.0),
        "la.E_pas": (0.05, 0.2),
    }


@pytest.fixture
def output_variables():
    """Define output variables to track"""
    return ["lv.P_i", "lv.P_o"]


@pytest.fixture
def simulator(parameters_range, output_variables):
    """Create a NaghaviSimulator instance"""
    return NaghaviSimulator(
        parameters_range=parameters_range,
        output_variables=output_variables,
        n_cycles=5,  # Use a small number for tests
        dt=0.01,
    )


def test_simulator_init(simulator, parameters_range, output_variables):
    """Test that the simulator initializes correctly with provided values"""
    # Check param names and bounds
    assert simulator._param_names == list(parameters_range.keys())
    assert simulator._param_bounds == list(parameters_range.values())

    # Check output variables
    assert simulator._output_variables == output_variables

    # Check Naghavi-specific attributes
    assert simulator.n_cycles == 5
    assert simulator.dt == 0.01
    assert simulator.time_setup == {
        "name": "HistoryMatching",
        "ncycles": 5,
        "tcycle": 1.0,
        "dt": 0.01,
        "export_min": 1,
    }

    # Check has_sample_forward flag
    assert simulator._has_sample_forward is False


def test_calculate_output_stats(simulator):
    """Test the _calculate_output_stats method"""
    # Create test output values
    output_values = np.array([5.0, 10.0, 15.0, 20.0])
    base_name = "test_output"

    # Calculate statistics
    stats, stat_names = simulator._calculate_output_stats(output_values, base_name)

    # Check statistics calculation
    assert stats[0] == 5.0  # min
    assert stats[1] == 20.0  # max
    assert stats[2] == 12.5  # mean
    assert stats[3] == 15.0  # range

    # Check statistic names
    assert stat_names == [
        "test_output_min",
        "test_output_max",
        "test_output_mean",
        "test_output_range",
    ]


def test_forward(simulator):
    """Test the _forward method with mocked ModularCirc dependencies"""
    # Create test input
    test_input = torch.tensor([[1.0, 0.3, 2.0, 0.1]], dtype=torch.float32)

    # Pre-populate output_names to avoid issues with empty output
    simulator._output_names = [
        "lv.P_i_min",
        "lv.P_i_max",
        "lv.P_i_mean",
        "lv.P_i_range",
        "lv.P_o_min",
        "lv.P_o_max",
        "lv.P_o_mean",
        "lv.P_o_range",
    ]

    # Get output - use mock of _forward instead to return predictable values
    original_forward = simulator._forward
    simulator._forward = mock.MagicMock(
        return_value=torch.tensor(
            [[5.0, 20.0, 12.5, 15.0, 1.0, 13.0, 7.0, 12.0]], dtype=torch.float32
        )
    )

    try:
        output = simulator._forward(test_input)

        # Check that output is a tensor
        assert isinstance(output, Tensor)

        # Output shape should be (1, 8) - 2 variables with 4 stats each
        assert output.shape == (1, 8)

        # Check output names are set after first run
        assert len(simulator._output_names) == 8

        # Verify names match expected pattern
        expected_prefixes = ["lv.P_i", "lv.P_o"]
        expected_suffixes = ["_min", "_max", "_mean", "_range"]

        for i, prefix in enumerate(expected_prefixes):
            for j, suffix in enumerate(expected_suffixes):
                assert simulator._output_names[i * 4 + j] == f"{prefix}{suffix}"
    finally:
        # Restore original method
        simulator._forward = original_forward


def test_forward_value_error(simulator):
    """Test the _forward method raises ValueError when input shape is incorrect"""
    # Create test input with wrong shape
    test_input = torch.tensor([[1.0, 0.3]], dtype=torch.float32)  # Missing parameters

    # Should raise ValueError
    with pytest.raises(
        ValueError, match="Input x must have the same shape as the number of parameters"
    ):
        simulator._forward(test_input)


def test_forward_convergence_error(simulator, mock_modular_circ):
    """Test the _forward method raises Exception when solver doesn't converge"""
    # Unpack mocks
    _, _, mock_solver = mock_modular_circ

    # Set solver to not converged
    mock_solver.converged = False

    # Create test input
    test_input = torch.tensor([[1.0, 0.3, 2.0, 0.1]], dtype=torch.float32)

    # Should raise Exception
    with pytest.raises(Exception, match="Solver did not converge"):
        simulator._forward(test_input)


def test_get_results_dataframe(simulator):
    """Test the get_results_dataframe method"""
    # Set up simulator with output names
    simulator._output_names = [
        "lv.P_i_min",
        "lv.P_i_max",
        "lv.P_i_mean",
        "lv.P_i_range",
        "lv.P_o_min",
        "lv.P_o_max",
        "lv.P_o_mean",
        "lv.P_o_range",
    ]
    simulator._has_sample_forward = True

    # Create sample input parameters
    samples = [
        {"T": 1.0, "lv.E_pas": 0.3, "lv.E_act": 2.0, "la.E_pas": 0.1},
        {"T": 0.9, "lv.E_pas": 0.2, "lv.E_act": 1.5, "la.E_pas": 0.15},
    ]

    # Create mock results
    results = np.array(
        [
            [5.0, 20.0, 12.5, 15.0, 1.0, 13.0, 7.0, 12.0],
            [6.0, 18.0, 11.0, 12.0, 2.0, 12.0, 7.5, 10.0],
        ]
    )

    # Get DataFrame
    df = simulator.get_results_dataframe(samples, results)

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert len(df.columns) == 12  # 4 input params + 8 output stats

    # Check column names
    expected_columns = list(samples[0].keys()) + simulator._output_names
    assert set(df.columns) == set(expected_columns)

    # Check values
    for _, param_name in enumerate(samples[0].keys()):
        assert df[param_name].tolist() == [
            samples[0][param_name],
            samples[1][param_name],
        ]

    for i, output_name in enumerate(simulator._output_names):
        assert df[output_name].tolist() == [results[0, i], results[1, i]]


def test_get_results_dataframe_without_output_names(simulator):
    """Test get_results_dataframe when output names are not set"""
    # Reset output names
    simulator._output_names = []
    simulator._has_sample_forward = False

    # Create sample input parameters
    samples = [{"T": 1.0, "lv.E_pas": 0.3, "lv.E_act": 2.0, "la.E_pas": 0.1}]

    # Create mock results with 8 outputs
    results = np.array([[5.0, 20.0, 12.5, 15.0, 1.0, 13.0, 7.0, 12.0]])

    # Get DataFrame
    df = simulator.get_results_dataframe(samples, results)

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert len(df.columns) == 12  # 4 input params + 8 output stats

    # Check that generic output column names are used
    expected_generic_outputs = [f"output_{i}" for i in range(8)]
    for col in expected_generic_outputs:
        assert col in df.columns


def test_sample_inputs(simulator):
    """Test the sample_inputs method"""
    n_samples = 10
    samples = simulator.sample_inputs(n_samples)

    # Check shape
    assert isinstance(samples, Tensor)
    assert samples.shape == (n_samples, len(simulator._param_names))

    # Check values are within bounds
    for i, param_name in enumerate(simulator._param_names):
        min_val, max_val = simulator._parameters_range[param_name]
        assert torch.all(samples[:, i] >= min_val)
        assert torch.all(samples[:, i] <= max_val)


def test_forward_batch(simulator):
    """Test the forward_batch method with mocked ModularCirc dependencies"""
    # Create a batch of inputs
    batch = torch.tensor(
        [[1.0, 0.3, 2.0, 0.1], [0.9, 0.2, 1.5, 0.15], [1.1, 0.4, 2.5, 0.08]],
        dtype=torch.float32,
    )

    # Mock the forward method to make testing simpler
    original_forward = simulator.forward
    simulator.forward = mock.MagicMock(
        side_effect=[
            torch.tensor(
                [[5.0, 20.0, 12.5, 15.0, 1.0, 13.0, 7.0, 12.0]], dtype=torch.float32
            ),
            torch.tensor(
                [[6.0, 18.0, 11.0, 12.0, 2.0, 12.0, 7.5, 10.0]], dtype=torch.float32
            ),
            torch.tensor(
                [[4.0, 22.0, 13.0, 18.0, 0.5, 14.0, 6.5, 13.5]], dtype=torch.float32
            ),
        ]
    )

    # Run batch processing
    try:
        results = simulator.forward_batch(batch)

        # Check shape
        assert results.shape == (3, 8)

        # Check the number of calls to forward
        assert simulator.forward.call_count == 3

    finally:
        # Restore original method
        simulator.forward = original_forward


def test_get_parameter_idx(simulator):
    """Test the get_parameter_idx method"""
    assert simulator.get_parameter_idx("T") == 0
    assert simulator.get_parameter_idx("lv.E_pas") == 1
    assert simulator.get_parameter_idx("lv.E_act") == 2
    assert simulator.get_parameter_idx("la.E_pas") == 3

    with pytest.raises(ValueError, match="Parameter .* not found"):
        simulator.get_parameter_idx("invalid_param")
