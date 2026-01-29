"""Tests for Bayesian evidence computation."""

import pyro
import pyro.distributions as dist
import pytest
import torch
from autoemulate.calibration.bayes import (
    BayesianCalibration,
    extract_log_probabilities,
)
from autoemulate.calibration.evidence import EvidenceComputation
from autoemulate.core.types import TensorLike
from autoemulate.emulators.gaussian_process.exact import GaussianProcess
from autoemulate.simulations.epidemic import Epidemic
from autoemulate.simulations.projectile import Projectile
from pyro.infer import MCMC
from pyro.infer.mcmc import RandomWalkKernel


@pytest.fixture
def simple_mcmc_setup():
    """Create a simple MCMC setup for testing."""
    sim = Epidemic(log_level="error")
    x = sim.sample_inputs(50)
    y, _ = sim.forward_batch(x)
    assert isinstance(y, TensorLike)

    gp = GaussianProcess(x, y)
    gp.fit(x, y)

    # Create observations
    observations = {"distance": y[:5, 0]}
    bc = BayesianCalibration(
        gp, sim.parameters_range, observations, 1.0, log_level="error"
    )

    # Run MCMC with sufficient samples for evidence computation
    mcmc = bc.run_mcmc(warmup_steps=50, num_samples=500, num_chains=10)

    return mcmc, bc.model, sim


class TestExtractLogProbabilities:
    """Tests for extract_log_probabilities function."""

    def test_basic_extraction(self, simple_mcmc_setup):
        """Test basic log probability extraction."""
        mcmc, model, _ = simple_mcmc_setup

        samples, log_probs = extract_log_probabilities(mcmc, model)

        # Check shapes
        assert samples.ndim == 3  # (chains, samples_per_chain, ndim)
        assert log_probs.ndim == 2  # (chains, samples_per_chain)
        assert samples.shape[0] == 10  # num_chains
        assert samples.shape[1] == 500  # num_samples_per_chain
        assert samples.shape[2] == 2  # num_parameters (c, v0)
        assert log_probs.shape == (10, 500)

    def test_log_probs_are_finite(self, simple_mcmc_setup):
        """Test that log probabilities are finite."""
        mcmc, model, _ = simple_mcmc_setup

        samples, log_probs = extract_log_probabilities(mcmc, model)

        # Check no NaN or Inf
        assert not torch.isnan(log_probs).any()
        assert not torch.isinf(log_probs).any()

    def test_device_handling(self, simple_mcmc_setup):
        """Test device parameter handling."""
        mcmc, model, _ = simple_mcmc_setup

        # Test with CPU device
        samples_cpu, log_probs_cpu = extract_log_probabilities(
            mcmc, model, device="cpu"
        )
        assert samples_cpu.device.type == "cpu"
        assert log_probs_cpu.device.type == "cpu"

        # Test with default device (None)
        samples_default, log_probs_default = extract_log_probabilities(
            mcmc, model, device=None
        )
        assert samples_default.device.type == "cpu"
        assert log_probs_default.device.type == "cpu"

    def test_empty_mcmc_raises_error(self):
        """Test that empty MCMC object raises ValueError."""

        def dummy_model():
            pyro.sample("x", dist.Normal(0, 1))

        kernel = RandomWalkKernel(dummy_model)
        mcmc = MCMC(kernel, num_samples=0, warmup_steps=0, num_chains=1)

        with pytest.raises(
            ValueError, match="(Failed to extract samples|contains no samples)"
        ):
            extract_log_probabilities(mcmc, dummy_model)


class TestEvidenceComputation:
    """Tests for EvidenceComputation class."""

    def test_initialization(self, simple_mcmc_setup):
        """Test EvidenceComputation initialization."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(
            mcmc, model, training_proportion=0.5, temperature=0.8, log_level="error"
        )

        assert ec.mcmc is mcmc
        assert ec.model is model
        assert ec.training_proportion == 0.5
        assert ec.temperature == 0.8
        assert ec.samples is not None
        assert ec.log_probs is not None
        assert ec.samples.shape[0] == 10  # num_chains
        assert ec.samples.shape[1] == 500  # num_samples_per_chain

    def test_invalid_training_proportion(self, simple_mcmc_setup):
        """Test that invalid training_proportion raises ValueError."""
        mcmc, model, _ = simple_mcmc_setup

        with pytest.raises(ValueError, match="training_proportion must be between"):
            EvidenceComputation(mcmc, model, training_proportion=0.0)

        with pytest.raises(ValueError, match="training_proportion must be between"):
            EvidenceComputation(mcmc, model, training_proportion=1.0)

        with pytest.raises(ValueError, match="training_proportion must be between"):
            EvidenceComputation(mcmc, model, training_proportion=-0.5)

    def test_invalid_temperature(self, simple_mcmc_setup):
        """Test that invalid temperature raises ValueError."""
        mcmc, model, _ = simple_mcmc_setup

        with pytest.raises(ValueError, match="temperature must be positive"):
            EvidenceComputation(mcmc, model, temperature=0.0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            EvidenceComputation(mcmc, model, temperature=-1.0)

    def test_unsupported_flow_model(self, simple_mcmc_setup):
        """Test that unsupported flow_model raises ValueError."""
        mcmc, model, _ = simple_mcmc_setup

        with pytest.raises(ValueError, match="Unsupported flow_model"):
            EvidenceComputation(mcmc, model, flow_model="UnsupportedModel")

    def test_training_proportion_too_small_for_chains(self, simple_mcmc_setup):
        """Test that training_proportion too small for num_chains raises ValueError."""
        mcmc, model, _ = simple_mcmc_setup

        # With 10 chains, training_proportion=0.05 would give 0 training chains
        with pytest.raises(
            ValueError, match="training_proportion.*too small.*training chains"
        ):
            EvidenceComputation(mcmc, model, training_proportion=0.05)

    def test_run_basic(self, simple_mcmc_setup):
        """Test basic evidence computation."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(mcmc, model, log_level="error")
        results = ec.run(epochs=50, verbose=False)

        # Check results structure
        assert isinstance(results, dict)
        assert "ln_evidence" in results
        assert "ln_inv_evidence" in results
        assert "error_lower" in results
        assert "error_upper" in results
        assert "samples_shape" in results
        assert "num_chains" in results
        assert "num_samples_per_chain" in results
        assert "num_parameters" in results

        # Check values are finite
        assert not torch.isnan(torch.tensor(results["ln_evidence"]))
        assert not torch.isinf(torch.tensor(results["ln_evidence"]))

        # Check relationship: ln_evidence = -ln_inv_evidence
        assert abs(results["ln_evidence"] + results["ln_inv_evidence"]) < 1e-6

        # Check dimensions
        assert results["num_chains"] == 10
        assert results["num_samples_per_chain"] == 500
        assert results["num_parameters"] == 2

    @pytest.mark.parametrize("training_proportion", [0.3, 0.5, 0.7])
    def test_different_training_proportions(
        self, simple_mcmc_setup, training_proportion
    ):
        """Test evidence computation with different training proportions."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(
            mcmc, model, training_proportion=training_proportion, log_level="error"
        )
        results = ec.run(epochs=5, verbose=False)

        assert isinstance(results, dict)
        assert "ln_evidence" in results

    @pytest.mark.parametrize("temperature", [0.5, 0.8, 1.0])
    def test_different_temperatures(self, simple_mcmc_setup, temperature):
        """Test evidence computation with different temperatures."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(
            mcmc, model, temperature=temperature, log_level="error"
        )
        results = ec.run(epochs=5, verbose=False)

        assert isinstance(results, dict)
        assert "ln_evidence" in results

    def test_get_chains_before_compute(self, simple_mcmc_setup):
        """Test that get_chains raises error before run."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(mcmc, model, log_level="error")

        with pytest.raises(RuntimeError, match="Call run"):
            ec.get_chains()

    def test_get_flow_model_before_compute(self, simple_mcmc_setup):
        """Test that get_flow_model raises error before run."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(mcmc, model, log_level="error")

        with pytest.raises(RuntimeError, match="Call run"):
            ec.get_flow_model()

    def test_get_evidence_object_before_compute(self, simple_mcmc_setup):
        """Test that get_evidence_object raises error before run."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(mcmc, model, log_level="error")

        with pytest.raises(RuntimeError, match="Call run"):
            ec.get_evidence_object()

    def test_getter_methods_after_compute(self, simple_mcmc_setup):
        """Test getter methods work after run."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(mcmc, model, log_level="error")
        ec.run(epochs=5, verbose=False)

        # Test all getter methods
        chains = ec.get_chains()
        assert chains is not None

        flow = ec.get_flow_model()
        assert flow is not None

        evidence = ec.get_evidence_object()
        assert evidence is not None

    def test_split_data_method(self, simple_mcmc_setup):
        """Test split_data method separately."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(
            mcmc, model, training_proportion=0.6, log_level="error"
        )
        chains_train, chains_infer = ec.split_data()

        # Check that chains were created and split
        assert ec.chains is not None
        assert chains_train is not None
        assert chains_infer is not None
        assert ec.chains_train is chains_train
        assert ec.chains_infer is chains_infer

        # Check split proportion (approximately)
        total_samples = ec.chains.samples.shape[0]
        train_samples = chains_train.samples.shape[0]
        train_ratio = train_samples / total_samples
        assert 0.55 < train_ratio < 0.65  # Allow some tolerance

    def test_fit_flow_method(self, simple_mcmc_setup):
        """Test fit_flow method separately."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(mcmc, model, log_level="error")
        ec.split_data()
        ec.fit_flow(epochs=10, verbose=False)

        # Check that flow was created and trained
        assert ec.flow is not None
        # Check flow has expected attributes from RQSplineModel
        assert hasattr(ec.flow, "fit")
        assert hasattr(ec.flow, "temperature")

    def test_fit_flow_without_split_raises_error(self, simple_mcmc_setup):
        """Test that fit_flow raises error if split_data not called."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(mcmc, model, log_level="error")

        with pytest.raises(RuntimeError, match="Must call split_data"):
            ec.fit_flow(epochs=10)

    def test_compute_ln_evidence_method(self, simple_mcmc_setup):
        """Test compute_ln_evidence method separately."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(mcmc, model, log_level="error")
        ec.split_data()
        ec.fit_flow(epochs=10, verbose=False)
        results = ec.compute_ln_evidence()

        # Check results structure
        assert isinstance(results, dict)
        assert "ln_evidence" in results
        assert "ln_inv_evidence" in results
        assert "error_lower" in results
        assert "error_upper" in results
        assert results["num_chains"] == 10
        assert results["num_parameters"] == 2

    def test_compute_ln_evidence_without_fit_raises_error(self, simple_mcmc_setup):
        """Test that compute_ln_evidence raises error if fit_flow not called."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(mcmc, model, log_level="error")
        ec.split_data()

        with pytest.raises(RuntimeError, match="Must call split_data.*and fit_flow"):
            ec.compute_ln_evidence()

    def test_modular_workflow(self, simple_mcmc_setup):
        """Test the modular workflow: split -> fit -> compute."""
        mcmc, model, _ = simple_mcmc_setup

        ec = EvidenceComputation(mcmc, model, log_level="error")

        # Step 1: Split data
        chains_train, chains_infer = ec.split_data()
        assert chains_train is not None
        assert chains_infer is not None

        # Step 2: Fit flow
        ec.fit_flow(epochs=10, verbose=False)
        assert ec.flow is not None

        # Step 3: Compute evidence
        results = ec.compute_ln_evidence()
        assert "ln_evidence" in results

        # Verify consistency with convenience method
        ec2 = EvidenceComputation(mcmc, model, log_level="error")
        results2 = ec2.run(epochs=10, verbose=False)

        # Both should produce similar results (may not be identical due to randomness)
        assert abs(results["ln_evidence"] - results2["ln_evidence"]) < 5.0


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self):
        """Test complete workflow from calibration to evidence computation."""
        # Setup simulation
        sim = Projectile()
        x = sim.sample_inputs(50)
        y, _ = sim.forward_batch(x)

        # Train emulator
        gp = GaussianProcess(x, y)
        gp.fit(x, y)

        # Run calibration
        observations = {"distance": y[:5, 0]}
        bc = BayesianCalibration(
            gp, sim.parameters_range, observations, 1.0, log_level="error"
        )
        mcmc = bc.run_mcmc(warmup_steps=50, num_samples=100, num_chains=10)

        # Compute evidence
        ec = EvidenceComputation(mcmc, bc.model, log_level="error")
        results = ec.run(epochs=5, verbose=False)

        # Verify results
        assert "ln_evidence" in results
        assert isinstance(results["ln_evidence"], float)
        assert not torch.isnan(torch.tensor(results["ln_evidence"]))

    def test_custom_flow_kwargs_rqspline(self, simple_mcmc_setup):
        """Test evidence computation with custom RQSpline flow parameters."""
        mcmc, model, _ = simple_mcmc_setup

        # Custom parameters for RQSpline
        flow_kwargs = {
            "n_layers": 12,
            "n_bins": 16,
            "hidden_size": [128, 128],
            "learning_rate": 0.0005,
        }

        ec = EvidenceComputation(
            mcmc, model, flow_kwargs=flow_kwargs, log_level="error"
        )
        results = ec.run(epochs=10, verbose=False)

        # Verify results are valid
        assert "ln_evidence" in results
        assert isinstance(results["ln_evidence"], float)
        assert not torch.isnan(torch.tensor(results["ln_evidence"]))
        assert not torch.isinf(torch.tensor(results["ln_evidence"]))

        # Verify flow was created with custom parameters
        assert ec.flow is not None

    def test_flow_kwargs_override_defaults(self, simple_mcmc_setup):
        """Test that flow_kwargs override default parameters."""
        mcmc, model, _ = simple_mcmc_setup

        # Override standardize (default is True)
        flow_kwargs = {"standardize": False, "n_layers": 4}

        ec = EvidenceComputation(
            mcmc, model, flow_kwargs=flow_kwargs, log_level="error"
        )

        # Initialize to access flow creation
        ec._extract_and_validate_samples()
        ec._initialize_harmonic_components()
        ec.split_data()

        # Create flow and check it was configured
        flow = ec._create_flow_model(ec.ndim)
        assert flow is not None

        # Run full computation to ensure it works
        results = ec.run(epochs=10, verbose=False)
        assert "ln_evidence" in results
        assert isinstance(results["ln_evidence"], float)
