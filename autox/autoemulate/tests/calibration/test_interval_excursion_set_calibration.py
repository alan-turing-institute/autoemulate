import pytest
from autoemulate.calibration.interval_excursion_set import (
    IntervalExcursionSetCalibration,
)
from autoemulate.core.types import TensorLike
from autoemulate.emulators.gaussian_process.exact import GaussianProcess
from autoemulate.simulations.projectile import Projectile, ProjectileMultioutput

TEST_BOUNDS_1D = {"distance": (30_000, 40_000)}
TEST_BOUNDS_2D = {"distance": (30_000, 40_000), "impact_velocity": (400, 600)}
N_SAMPLES = 10


@pytest.mark.parametrize(
    ("n_chains", "output_bounds"),
    [
        (1, TEST_BOUNDS_1D),
        (2, TEST_BOUNDS_1D),
    ],
)
def test_single_output(n_chains, output_bounds):
    """
    Test HMC with single output.
    """
    sim = Projectile()
    x = sim.sample_inputs(100)
    y, _ = sim.forward_batch(x)
    assert isinstance(y, TensorLike)
    gp = GaussianProcess(x, y)
    gp.fit(x, y)

    iesc = IntervalExcursionSetCalibration(
        gp,
        sim.parameters_range,
        output_bounds=output_bounds,
        output_names=sim.output_names,
    )

    # check samples are generated
    mcmc = iesc.run_mcmc(
        warmup_steps=N_SAMPLES, num_samples=N_SAMPLES, num_chains=n_chains
    )
    samples = mcmc.get_samples(group_by_chain=True)
    assert "c" in samples
    assert "v0" in samples
    assert samples["c"].shape[0] == n_chains
    assert samples["c"].shape[1] == N_SAMPLES
    assert samples["v0"].shape[0] == n_chains
    assert samples["v0"].shape[1] == N_SAMPLES

    # posterior predictive
    pp = iesc.posterior_predictive(mcmc)
    assert "distance" in pp
    assert pp["distance"].shape[0] == N_SAMPLES * n_chains

    # Test SMC
    az_data = iesc.run_smc(n_particles=N_SAMPLES, return_az_data=True)
    post = az_data.posterior.to_dataframe()  # type: ignore  # noqa: PGH003
    assert post.shape == (N_SAMPLES, 2)
    assert "c" in post.columns
    assert "v0" in post.columns


@pytest.mark.parametrize(
    ("n_chains", "output_bounds"),
    [
        (1, TEST_BOUNDS_2D),
        (2, TEST_BOUNDS_2D),
    ],
)
def test_multi_output(n_chains, output_bounds):
    """
    Test HMC with multiple output.
    """
    sim = ProjectileMultioutput()
    x = sim.sample_inputs(100)
    y, _ = sim.forward_batch(x)
    assert isinstance(y, TensorLike)
    gp = GaussianProcess(x, y)
    gp.fit(x, y)

    iesc = IntervalExcursionSetCalibration(
        gp,
        sim.parameters_range,
        output_bounds=output_bounds,
        output_names=sim.output_names,
    )

    # check samples are generated
    mcmc = iesc.run_mcmc(
        warmup_steps=N_SAMPLES, num_samples=N_SAMPLES, num_chains=n_chains
    )
    samples = mcmc.get_samples(group_by_chain=True)
    assert "c" in samples
    assert "v0" in samples
    assert samples["c"].shape[0] == n_chains
    assert samples["c"].shape[1] == N_SAMPLES
    assert samples["v0"].shape[0] == n_chains
    assert samples["v0"].shape[1] == N_SAMPLES

    # posterior predictive
    pp = iesc.posterior_predictive(mcmc)
    assert "distance" in pp
    assert "impact_velocity" in pp
    assert pp["distance"].shape[0] == N_SAMPLES * n_chains
    assert pp["impact_velocity"].shape[0] == N_SAMPLES * n_chains

    # Test SMC
    az_data = iesc.run_smc(n_particles=N_SAMPLES, return_az_data=True)
    post = az_data.posterior.to_dataframe()  # type: ignore  # noqa: PGH003
    assert post.shape == (N_SAMPLES, 2)
    assert "c" in post.columns
    assert "v0" in post.columns
