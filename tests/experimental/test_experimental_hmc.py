from autoemulate.experimental.calibration.hmc import HMCCalibrator
from autoemulate.experimental.emulators.gaussian_process.exact import (
    GaussianProcessExact,
)
from autoemulate.experimental.simulations.projectile import (
    Projectile,
    ProjectileMultioutput,
)


def test_hmc_single_output():
    """
    Test HMC with single output (single observation).
    """
    sim = Projectile()
    x = sim.sample_inputs(100)
    y = sim.forward_batch(x)
    gp = GaussianProcessExact(x, y)
    gp.fit(x, y)

    # pick the first sim output as an observation
    observations = {"distance": y[0]}
    hmc = HMCCalibrator(gp, sim.parameters_range, observations, 1.0)
    assert hmc.observation_noise == {"distance": 1.0}

    # check samples are generated
    mcmc = hmc.run(warmup_steps=5, num_samples=5)
    samples = mcmc.get_samples()
    assert "c" in samples
    assert "v0" in samples
    assert samples["c"].shape[0] == 5

    # posterior predictive
    pp = hmc.posterior_predictive(mcmc)
    assert isinstance(pp, dict)
    pp = dict(pp)  # keeping type checker happy
    assert "distance" in pp
    # get a prediction for each mcmc sample
    assert pp["distance"].shape[0] == 5


def test_hmc_multiple_output():
    """
    Test HMC with multiple outputs (single observation per output).
    """
    sim = ProjectileMultioutput()
    x = sim.sample_inputs(100)
    y = sim.forward_batch(x)
    gp = GaussianProcessExact(x, y)
    gp.fit(x, y)

    # pick the first sim output as an observation
    observations = {
        "distance": y[:1, 0],
        "impact_velocity": y[:1, 1],
    }
    hmc = HMCCalibrator(gp, sim.parameters_range, observations, 1.0)
    assert hmc.observation_noise == {"distance": 1.0, "impact_velocity": 1.0}

    # check samples are generated
    mcmc = hmc.run(warmup_steps=5, num_samples=5)
    samples = mcmc.get_samples()
    assert "c" in samples
    assert "v0" in samples
    assert samples["c"].shape[0] == 5

    # posterior predictive
    pp = hmc.posterior_predictive(mcmc)
    assert isinstance(pp, dict)
    pp = dict(pp)  # keeping type checker happy
    assert "distance" in pp
    assert "impact_velocity" in pp
    # get a prediction per output and per mcmc sample
    assert pp["distance"].shape[0] == 5
    assert pp["impact_velocity"].shape[0] == 5


def test_hmc_multiple_obs():
    """
    Test HMC with multiple observations per output.
    """
    sim = ProjectileMultioutput()
    x = sim.sample_inputs(100)
    y = sim.forward_batch(x)
    gp = GaussianProcessExact(x, y)
    gp.fit(x, y)

    # pick the first 10 sim outputs as observations
    observations = {
        "distance": y[:10, 0],
        "impact_velocity": y[:10, 1],
    }
    hmc = HMCCalibrator(
        gp,
        sim.parameters_range,
        observations,
        observation_noise={"distance": 20.0, "impact_velocity": 10.0},
    )
    assert hmc.observation_noise == {"distance": 20.0, "impact_velocity": 10.0}

    # check samples are generated
    mcmc = hmc.run(warmup_steps=5, num_samples=5)
    samples = mcmc.get_samples()
    assert "c" in samples
    assert "v0" in samples
    assert samples["c"].shape[0] == 5

    # posterior predictive
    pp = hmc.posterior_predictive(mcmc)
    assert isinstance(pp, dict)
    pp = dict(pp)  # keeping type checker happy
    assert "distance" in pp
    assert "impact_velocity" in pp
    # the pp samples are now shape [n_mcmc_samples, n_observations]
    assert pp["distance"].shape[0] == 5
    assert pp["distance"].shape[1] == 10
    assert pp["impact_velocity"].shape[0] == 5
    assert pp["impact_velocity"].shape[1] == 10
