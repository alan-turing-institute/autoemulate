import pytest
from autoemulate.experimental.calibration.bayes import BayesianCalibration
from autoemulate.experimental.emulators.gaussian_process.exact import (
    GaussianProcess,
)
from autoemulate.experimental.simulations.projectile import (
    Projectile,
    ProjectileMultioutput,
)
from autoemulate.experimental.types import TensorLike


@pytest.mark.parametrize(
    ("n_obs", "n_chains", "n_samples"),
    [(1, 1, 10), (10, 1, 10), (1, 2, 10), (10, 2, 10)],
)
def test_hmc_single_output(n_obs, n_chains, n_samples):
    """
    Test HMC with single output.
    """
    sim = Projectile()
    x = sim.sample_inputs(100)
    y = sim.forward_batch(x)
    assert isinstance(y, TensorLike)
    gp = GaussianProcess(x, y)
    gp.fit(x, y)

    # pick the first n_obs sim outputs as observations
    observations = {"distance": y[:n_obs, 0]}
    bc = BayesianCalibration(gp, sim.parameters_range, observations, 1.0)
    assert bc.observation_noise == {"distance": 1.0}

    # check samples are generated
    mcmc = bc.run_mcmc(warmup_steps=10, num_samples=n_samples, num_chains=n_chains)
    samples = mcmc.get_samples(group_by_chain=True)
    assert "c" in samples
    assert "v0" in samples
    assert samples["c"].shape[0] == n_chains
    assert samples["c"].shape[1] == n_samples
    assert samples["v0"].shape[0] == n_chains
    assert samples["v0"].shape[1] == n_samples

    # posterior predictive
    pp = bc.posterior_predictive(mcmc)
    assert isinstance(pp, dict)
    pp = dict(pp)  # keeping type checker happy
    assert "distance" in pp
    # get a prediction per mcmc sample and observation
    assert pp["distance"].shape[0] == n_samples * n_chains
    assert pp["distance"].shape[1] == n_obs


@pytest.mark.parametrize(
    ("n_obs", "n_chains", "n_samples"),
    [(1, 1, 10), (10, 1, 10), (1, 2, 10), (10, 2, 10)],
)
def test_hmc_multiple_output(n_obs, n_chains, n_samples):
    """
    Test HMC with multiple outputs.
    """
    sim = ProjectileMultioutput()
    x = sim.sample_inputs(100)
    y = sim.forward_batch(x)
    assert isinstance(y, TensorLike)
    gp = GaussianProcess(x, y)
    gp.fit(x, y)

    # pick the first n_obs sim outputs as observations
    observations = {
        "distance": y[:n_obs, 0],
        "impact_velocity": y[:n_obs, 1],
    }
    bc = BayesianCalibration(gp, sim.parameters_range, observations, 1.0)
    assert bc.observation_noise == {"distance": 1.0, "impact_velocity": 1.0}

    # check samples are generated
    mcmc = bc.run_mcmc(warmup_steps=5, num_samples=n_samples, num_chains=n_chains)
    samples = mcmc.get_samples(group_by_chain=True)
    assert "c" in samples
    assert "v0" in samples
    assert samples["c"].shape[0] == n_chains
    assert samples["c"].shape[1] == n_samples
    assert samples["v0"].shape[0] == n_chains
    assert samples["v0"].shape[1] == n_samples

    # posterior predictive
    pp = bc.posterior_predictive(mcmc)
    assert isinstance(pp, dict)
    pp = dict(pp)  # keeping type checker happy
    assert "distance" in pp
    assert "impact_velocity" in pp
    # get a prediction per mcmc sample and observation
    assert pp["distance"].shape[0] == n_samples * n_chains
    assert pp["distance"].shape[1] == n_obs
    assert pp["impact_velocity"].shape[0] == n_samples * n_chains
    assert pp["impact_velocity"].shape[1] == n_obs
