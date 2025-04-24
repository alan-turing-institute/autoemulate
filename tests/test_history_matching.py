from copy import deepcopy
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from autoemulate.history_matching import HistoryMatcher
from autoemulate.simulations.base import Simulator


class MockSimulator(Simulator):
    """Mock simulator class for testing"""

    def __init__(self, param_names, output_names):
        # Create parameter ranges (0 to 1 for each parameter)
        param_bounds = {name: (0.0, 1.0) for name in param_names}

        # Call the parent class constructor
        super().__init__(param_bounds)

        # Set output names directly for testing
        self._output_names = output_names
        self._has_sample_forward = True

    def sample_inputs(self, n_samples):
        """Generate random parameter samples"""
        samples = []
        for _ in range(n_samples):
            sample = {name: np.random.uniform(0, 1) for name in self.param_names}
            samples.append(sample)
        return samples

    def sample_forward(self, params):
        """Run forward simulation with given parameters"""
        # Simple linear transformation for testing
        # Return values corresponding to each parameter
        return np.array([params[name] for name in self.param_names])


class TestHistoryMatcher:
    @pytest.fixture
    def simple_matcher(self):
        """Create a simple history matcher instance for testing"""
        param_names = ["x1", "x2"]
        output_names = ["y1", "y2"]
        simulator = MockSimulator(param_names, output_names)
        observations = {"y1": (0.5, 0.01), "y2": (0.7, 0.01)}  # (mean, variance)
        matcher = HistoryMatcher(simulator, observations, threshold=3.0)
        return matcher

    def test_init(self, simple_matcher):
        """Test constructor"""
        assert simple_matcher.threshold == 3.0
        assert simple_matcher.discrepancy == 0.0
        assert simple_matcher.rank == 1
        assert simple_matcher.observations == {"y1": (0.5, 0.01), "y2": (0.7, 0.01)}

    def test_init_value_error(self):
        """Test constructor raises error when observations don't match simulator outputs"""
        param_names = ["x1", "x2"]
        output_names = ["y1", "y2"]
        simulator = MockSimulator(param_names, output_names)

        # Observation keys that are not a subset of simulator output names
        observations = {
            "y1": (0.5, 0.01),
            "y3": (0.7, 0.01),  # y3 is not in simulator outputs
        }

        with pytest.raises(ValueError) as excinfo:
            HistoryMatcher(simulator, observations)

        assert "must be a subset of simulator output names" in str(excinfo.value)

    def test_calculate_implausibility(self, simple_matcher):
        """Test implausibility calculation"""
        predictions = {
            "y1": (0.6, 0.01),  # Close to observation (0.5, 0.01)
            "y2": (1.5, 0.01),  # Far from observation (0.7, 0.01)
        }

        result = simple_matcher.calculate_implausibility(predictions)

        # Check returned structure
        assert "I" in result
        assert "NROY" in result
        assert "RO" in result

        # Check implausibility values
        # I = |obs_mean - pred_mean| / sqrt(pred_var + discrepancy + obs_var)
        # For y1: |0.5 - 0.6| / sqrt(0.01 + 0.0 + 0.01) = 0.1 / sqrt(0.02) ≈ 0.71
        # For y2: |0.7 - 1.5| / sqrt(0.01 + 0.0 + 0.01) = 0.8 / sqrt(0.02) ≈ 5.66
        np.testing.assert_almost_equal(result["I"][0], 0.1 / np.sqrt(0.02), decimal=2)
        np.testing.assert_almost_equal(result["I"][1], 0.8 / np.sqrt(0.02), decimal=2)

        # y1 should be in NROY (implausibility ≈ 0.71 < threshold 3.0)
        # y2 should be in RO (implausibility ≈ 5.66 > threshold 3.0)
        assert 0 in result["NROY"]
        assert 1 in result["RO"]

    def test_run_wave_with_simulator(self, simple_matcher):
        """Test running a wave using the simulator"""
        # Create sample parameters
        params = [
            {"x1": 0.5, "x2": 0.7},  # Should be close to observations
            {"x1": 0.1, "x2": 0.2},  # Should be further from observations
        ]

        # Run the wave
        successful_samples, impl_scores = simple_matcher.run_wave(
            params, use_emulator=False
        )

        # Check results
        assert len(successful_samples) == 2
        assert impl_scores.shape == (2, 2)  # 2 samples x 2 outputs

    def test_run_wave_with_emulator(self, simple_matcher):
        """Test running a wave using an emulator"""
        # Create mock emulator
        mock_emulator = MagicMock()
        # Setup predict to return means and stds for two outputs
        mock_emulator.predict.return_value = (
            np.array([[0.5, 0.7]]),  # means - close to observations
            np.array([[0.1, 0.1]]),  # stds
        )

        # Create sample parameters
        params = [{"x1": 0.5, "x2": 0.7}]

        # Run the wave
        successful_samples, impl_scores = simple_matcher.run_wave(
            params, use_emulator=True, emulator=mock_emulator
        )

        # Check results
        assert len(successful_samples) == 1
        assert impl_scores.shape == (1, 2)  # 1 sample x 2 outputs
        assert mock_emulator.predict.called

    def test_generate_new_samples(self, simple_matcher):
        """Test generating new samples within NROY space"""
        # Create some NROY samples
        nroy_samples = [
            {"x1": 0.4, "x2": 0.6},
            {"x1": 0.5, "x2": 0.7},
            {"x1": 0.6, "x2": 0.8},
        ]

        # Generate new samples
        new_samples = simple_matcher.generate_new_samples(nroy_samples, n_samples=5)

        # Check results
        assert len(new_samples) == 5
        for sample in new_samples:
            assert "x1" in sample
            assert "x2" in sample
            assert 0.4 <= sample["x1"] <= 0.6  # Within bounds of NROY samples
            assert 0.6 <= sample["x2"] <= 0.8  # Within bounds of NROY samples

    def test_generate_new_samples_empty_nroy(self, simple_matcher):
        """Test generating new samples with empty NROY"""
        # Generate new samples with empty NROY
        new_samples = simple_matcher.generate_new_samples([], n_samples=5)

        # Should fall back to simulator.sample_inputs
        assert len(new_samples) == 5

    def test_run_history_matching(self, simple_matcher):
        """Test full history matching process"""
        # Create mock emulator - avoid comparing arrays directly
        mock_emulator = MagicMock()
        # Configure predict method to return appropriate data for single prediction
        mock_emulator.predict.return_value = (
            np.array([[0.5, 0.7]]),  # means
            np.array([[0.1, 0.1]]),  # stds
        )

        # Don't use actual arrays for X_train_ and y_train_ to avoid array comparison issues
        mock_emulator.X_train_ = np.array([[0.5, 0.7]])
        mock_emulator.y_train_ = np.array([[0.5, 0.7]])

        # Mock update_emulator to avoid array comparisons in assertions
        with patch.object(
            simple_matcher, "update_emulator", return_value=mock_emulator
        ) as mock_update:
            # Run history matching with mocked update_emulator
            # Use n_samples_per_wave=15 to ensure we have enough samples to trigger the emulator update
            (
                all_samples,
                all_impl_scores,
                updated_emulator,
            ) = simple_matcher.run_history_matching(
                n_waves=2,
                n_samples_per_wave=15,  # Increased from 5 to 15
                use_emulator=True,
                initial_emulator=mock_emulator,
            )

            # Check results - simple assertions that avoid array comparison
            assert len(all_samples) > 0, "No samples were returned"
            assert all_impl_scores.shape[1] == 2, "Expected 2 outputs"
            assert (
                updated_emulator is mock_emulator
            ), "Emulator was not returned correctly"

            # Verify update_emulator was called at least once
            assert mock_update.call_count > 0, "update_emulator was not called"

    def test_update_emulator(self, simple_matcher):
        """Test updating the emulator with new data"""
        # Create mock emulator with stored training data
        mock_emulator = MagicMock()
        mock_emulator.X_train_ = np.array([[0.3, 0.4], [0.5, 0.6]])
        mock_emulator.y_train_ = np.array([[0.3, 0.4], [0.5, 0.6]])

        # Create new samples and outputs
        new_samples = np.array([[0.7, 0.8]])
        new_outputs = np.array([[0.7, 0.8]])

        # Update emulator - proper mocking to check method calls
        with patch.object(
            mock_emulator, "update", create=True
        ) as mock_update, patch.object(
            mock_emulator, "partial_fit", create=True
        ) as mock_partial_fit, patch.object(
            mock_emulator, "fit", create=True
        ) as mock_fit:
            updated_emulator = simple_matcher.update_emulator(
                mock_emulator, new_samples, new_outputs
            )

            # Check if any of the methods was called
            methods_called = [
                mock_update.call_count > 0,
                mock_partial_fit.call_count > 0,
                mock_fit.call_count > 0,
            ]
            assert any(methods_called), "None of the expected methods were called"

            # Verify that X_train_ and y_train_ attributes exist
            assert hasattr(updated_emulator, "X_train_")
            assert hasattr(updated_emulator, "y_train_")

    def test_update_emulator_dict_input(self, simple_matcher):
        """Test updating the emulator with dictionary input"""
        # Create mock emulator
        mock_emulator = MagicMock()
        mock_emulator.X_train_ = np.array([[0.3, 0.4]])
        mock_emulator.y_train_ = np.array([0.3, 0.4])

        # Create new samples as dictionaries and outputs
        new_samples = [{"x1": 0.7, "x2": 0.8}]
        new_outputs = np.array([0.7, 0.8])

        # Update emulator
        updated_emulator = simple_matcher.update_emulator(
            mock_emulator, new_samples, new_outputs, include_previous_data=False
        )

        # Check results - should only use new data
        expected_X = np.array([[0.7, 0.8]])
        np.testing.assert_array_equal(updated_emulator.X_train_, expected_X)

    def test_update_emulator_refit_hyperparams(self, simple_matcher):
        """Test updating the emulator with refit_hyperparams=True"""
        # Create mock emulator
        mock_emulator = MagicMock()
        mock_emulator.X_train_ = np.array([[0.3, 0.4]])
        mock_emulator.y_train_ = np.array([0.3, 0.4])

        # Make fit method that doesn't compare arrays directly
        mock_emulator.fit = MagicMock()

        # Create new samples and outputs
        new_samples = np.array([[0.7, 0.8]])
        new_outputs = np.array([0.7, 0.8])

        # Update emulator with refit_hyperparams=True
        updated_emulator = simple_matcher.update_emulator(
            mock_emulator, new_samples, new_outputs, refit_hyperparams=True
        )

        # Just check that fit was called
        assert mock_emulator.fit.called, "fit method was not called"
