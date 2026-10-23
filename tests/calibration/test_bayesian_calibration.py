import pyro
import pytest
from autoemulate.calibration.bayes import BayesianCalibration
from autoemulate.core.types import TensorLike
from autoemulate.emulators.gaussian_process.exact import GaussianProcess
from autoemulate.simulations.projectile import Projectile, ProjectileMultioutput

N_TRAINING_POINTS = 30
N_MCMC_SAMPLES = 4
N_MCMC_WARMUP = 3
GP_EPOCHS = 5
HMC_STEP_SIZE = 0.1
HMC_TRAJECTORY_LENGTH = 0.1
SINGLE_OUTPUT_HMC_CASES = [
    (1, 1, False),
    (10, 1, False),
    (10, 2, True),
]
MULTI_OUTPUT_HMC_CASES = [
    (1, 1, False),
    (10, 1, True),
]


@pytest.fixture(scope="module")
def projectile_calibration_setup():
    """Train a small single-output emulator once for calibration tests."""
    pyro.set_rng_seed(0)
    sim = Projectile()
    x = sim.sample_inputs(N_TRAINING_POINTS)
    y, _ = sim.forward_batch(x)
    assert isinstance(y, TensorLike)
    gp = GaussianProcess(x, y, epochs=GP_EPOCHS)
    gp.fit(x, y)
    return sim, gp, y


@pytest.fixture(scope="module")
def projectile_multioutput_calibration_setup():
    """Train a small multi-output emulator once for calibration tests."""
    pyro.set_rng_seed(1)
    sim = ProjectileMultioutput()
    x = sim.sample_inputs(N_TRAINING_POINTS)
    y, _ = sim.forward_batch(x)
    assert isinstance(y, TensorLike)
    gp = GaussianProcess(x, y, epochs=GP_EPOCHS)
    gp.fit(x, y)
    return sim, gp, y


@pytest.mark.parametrize(
    ("n_obs", "n_chains", "model_uncertainty"),
    SINGLE_OUTPUT_HMC_CASES,
)
def test_hmc_single_output(
    projectile_calibration_setup, n_obs, n_chains, model_uncertainty
):
    """
    Test HMC with single output.
    """
    sim, gp, y = projectile_calibration_setup

    # pick the first n_obs sim outputs as observations
    observations = {"distance": y[:n_obs, 0]}
    bc = BayesianCalibration(
        gp, sim.parameters_range, observations, 1.0, model_uncertainty=model_uncertainty
    )
    assert bc.observation_noise == {"distance": 1.0}

    # check samples are generated
    pyro.set_rng_seed(100 + n_obs + n_chains + int(model_uncertainty))
    mcmc = bc.run_mcmc(
        warmup_steps=N_MCMC_WARMUP,
        num_samples=N_MCMC_SAMPLES,
        num_chains=n_chains,
        sampler="hmc",
        step_size=HMC_STEP_SIZE,
        trajectory_length=HMC_TRAJECTORY_LENGTH,
    )
    samples = mcmc.get_samples(group_by_chain=True)
    assert "c" in samples
    assert "v0" in samples
    assert samples["c"].shape[0] == n_chains
    assert samples["c"].shape[1] == N_MCMC_SAMPLES
    assert samples["v0"].shape[0] == n_chains
    assert samples["v0"].shape[1] == N_MCMC_SAMPLES

    # posterior predictive
    pp = bc.posterior_predictive(mcmc)
    assert isinstance(pp, dict)
    pp = dict(pp)  # keeping type checker happy
    assert "distance" in pp
    # get a prediction per mcmc sample and observation
    assert pp["distance"].shape[0] == N_MCMC_SAMPLES * n_chains
    assert pp["distance"].shape[1] == n_obs


@pytest.mark.parametrize(
    ("n_obs", "n_chains", "model_uncertainty"),
    MULTI_OUTPUT_HMC_CASES,
)
def test_hmc_multiple_output(
    projectile_multioutput_calibration_setup, n_obs, n_chains, model_uncertainty
):
    """
    Test HMC with multiple outputs.
    """
    sim, gp, y = projectile_multioutput_calibration_setup

    # pick the first n_obs sim outputs as observations
    observations = {
        "distance": y[:n_obs, 0],
        "impact_velocity": y[:n_obs, 1],
    }
    bc = BayesianCalibration(
        gp, sim.parameters_range, observations, 1.0, model_uncertainty=model_uncertainty
    )
    assert bc.observation_noise == {"distance": 1.0, "impact_velocity": 1.0}

    # check samples are generated
    pyro.set_rng_seed(200 + n_obs + n_chains + int(model_uncertainty))
    mcmc = bc.run_mcmc(
        warmup_steps=N_MCMC_WARMUP,
        num_samples=N_MCMC_SAMPLES,
        num_chains=n_chains,
        sampler="hmc",
        step_size=HMC_STEP_SIZE,
        trajectory_length=HMC_TRAJECTORY_LENGTH,
    )
    samples = mcmc.get_samples(group_by_chain=True)
    assert "c" in samples
    assert "v0" in samples
    assert samples["c"].shape[0] == n_chains
    assert samples["c"].shape[1] == N_MCMC_SAMPLES
    assert samples["v0"].shape[0] == n_chains
    assert samples["v0"].shape[1] == N_MCMC_SAMPLES

    # posterior predictive
    pp = bc.posterior_predictive(mcmc)
    assert isinstance(pp, dict)
    pp = dict(pp)  # keeping type checker happy
    assert "distance" in pp
    assert "impact_velocity" in pp
    # get a prediction per mcmc sample and observation
    assert pp["distance"].shape[0] == N_MCMC_SAMPLES * n_chains
    assert pp["distance"].shape[1] == n_obs
    assert pp["impact_velocity"].shape[0] == N_MCMC_SAMPLES * n_chains
    assert pp["impact_velocity"].shape[1] == n_obs
