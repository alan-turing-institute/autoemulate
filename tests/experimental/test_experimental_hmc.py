import torch

from autoemulate.experimental.calibration.hmc import HMCCalibrator
from autoemulate.experimental.emulators.gaussian_process.exact import (
    GaussianProcessExact,
)
from autoemulate.experimental.simulations.projectile import ProjectileMultioutput


def test_hmc_single_obs():
    """
    Test HMC with single observation per output.
    """
    sim = ProjectileMultioutput()
    x = sim.sample_inputs(100)
    y = sim.forward_batch(x)
    gp = GaussianProcessExact(x, y)
    gp.fit(x, y)

    # pick the first sim output as an observation
    observations = {
        "distance": y[:1, 0].tolist(),
        "impact_velocity": y[:1, 1].tolist(),
    }
    hmc = HMCCalibrator(gp, sim.parameters_range, observations, 1.0)

    # check observations and noise shapes
    assert torch.allclose(hmc.observations, y[:1, :].float())
    assert torch.allclose(hmc.obs_noise, torch.tensor([1.0, 1.0]))

    # check samples are generates
    mcmc = hmc.run(warmup_steps=5, num_samples=5)
    samples = mcmc.get_samples()
    assert "c" in samples
    assert "v0" in samples
    assert samples["c"].shape[0] == 5


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
        "distance": y[:10, 0].tolist(),
        "impact_velocity": y[:10, 1].tolist(),
    }
    hmc = HMCCalibrator(
        gp,
        sim.parameters_range,
        observations,
        observation_noise={"distance": 20.0, "impact_velocity": 10.0},
    )

    # check observations and noise shapes
    assert torch.allclose(hmc.observations, y[:10, :].float())
    assert torch.allclose(hmc.obs_noise, torch.tensor([20.0, 10.0]))

    # check samples are generates
    mcmc = hmc.run(warmup_steps=5, num_samples=5)
    samples = mcmc.get_samples()
    assert "c" in samples
    assert "v0" in samples
    assert samples["c"].shape[0] == 5
