import pytest
import torch
from autoemulate.calibration.history_matching import (
    HistoryMatching,
    HistoryMatchingWorkflow,
)
from autoemulate.core.device import SUPPORTED_DEVICES, check_torch_device_is_available
from autoemulate.core.types import TensorLike
from autoemulate.emulators.gaussian_process.exact import GaussianProcess
from autoemulate.simulations.epidemic import Epidemic
from tests.simulations.test_base_simulator import MockSimulator


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
def history_matcher(observations):
    """Fixture for a basic HistoryMatching instance."""
    return HistoryMatching(
        observations=observations,
        threshold=3.0,
        model_discrepancy=0.1,
        rank=1,
    )


def test_history_matcher_init(history_matcher):
    """Test initialization of HistoryMatching."""
    assert history_matcher.threshold == 3.0
    assert history_matcher.discrepancy == 0.1
    assert history_matcher.rank == 1
    assert history_matcher.obs_means.shape == (1, 2)
    assert history_matcher.obs_vars.shape == (1, 2)


def test_calculate_implausibility(history_matcher, observations):
    """Test implausibility calculation."""
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

    # the output shape is the same irrespective of rank
    history_matcher.rank = 2
    impl_scores = history_matcher.calculate_implausibility(pred_means, pred_vars)
    assert impl_scores.shape == (1, 2)


def test_get_indices(history_matcher):
    """Test NROY and RO indices vary with rank."""
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


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_run(device):
    """Test the full history matching workflow with Epidemic simulator."""
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")

    simulator = Epidemic()
    x = simulator.sample_inputs(10)
    y = simulator.forward_batch(x)
    assert isinstance(y, TensorLike)

    # Run history matching
    gp = GaussianProcess(x, y, device=device)
    gp.fit(x, y)

    observations = {"infection_rate": (0.3, 0.05)}

    hm = HistoryMatchingWorkflow(
        simulator=simulator,
        emulator=gp,
        observations=observations,
        threshold=3.0,
        model_discrepancy=0.1,
        rank=1,
        device=device,
    )

    # call run first time
    hm.run(n_simulations=5)

    # Check basic structure of results
    assert isinstance(hm.train_x, TensorLike)
    assert isinstance(hm.emulator, GaussianProcess)

    assert len(hm.train_x) == 5

    # can run again
    hm.run(n_simulations=5)

    # We should get results for all valid samples
    assert len(hm.train_x) == 5 * 2


def test_run_max_tries():
    """Run history matching with observations that return no NROY params."""
    simulator = Epidemic()
    x = simulator.sample_inputs(10)
    y = simulator.forward_batch(x)
    assert isinstance(y, TensorLike)

    # Run history matching
    gp = GaussianProcess(x, y)
    gp.fit(x, y)

    # Extreme values outside the range of what the simulator returns
    observations = {"infection_rate": (100.0, 1.0)}

    hm = HistoryMatchingWorkflow(
        simulator=simulator,
        emulator=gp,
        observations=observations,
        threshold=3.0,
        model_discrepancy=0.1,
        rank=1,
    )

    with pytest.raises(RuntimeError):
        hm.run(n_simulations=5)
    with pytest.raises(RuntimeError):
        hm.run(n_simulations=5)
