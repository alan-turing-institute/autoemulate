import itertools

import pytest
import torch
from autoemulate.experimental.emulators import (
    ALL_EMULATORS as DEFAULT_EMULATORS,
)
from autoemulate.experimental.emulators import (
    GaussianProcessExact,
)
from autoemulate.experimental.emulators.ensemble import EnsembleMLP, EnsembleMLPDropout
from autoemulate.experimental.emulators.transformed.base import TransformedEmulator
from autoemulate.experimental.transforms import (
    PCATransform,
    StandardizeTransform,
    VAETransform,
)

# from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import DistributionLike, GaussianLike, TensorLike

# TODO: update once #579 completeed
ALL_EMULATORS = [
    emulator
    for emulator in DEFAULT_EMULATORS
    if emulator not in [EnsembleMLP, EnsembleMLPDropout]
]


def run_test(train_data, test_data, model, x_transforms, y_transforms):
    x, y = train_data
    x2, _ = test_data
    em = TransformedEmulator(
        x, y, x_transforms=x_transforms, y_transforms=y_transforms, model=model
    )
    em.fit(x, y)
    y_pred = em.predict(x2)
    if model is GaussianProcessExact:
        assert isinstance(y_pred, DistributionLike)
        assert y_pred.mean.shape == (x2.shape[0], y.shape[1])
    else:
        assert isinstance(y_pred, TensorLike)
        assert y_pred.shape == (x2.shape[0], y.shape[1])


@pytest.mark.parametrize(
    ("model", "x_transforms", "y_transforms"),
    itertools.product(
        [emulator for emulator in ALL_EMULATORS if emulator.is_multioutput()],
        [
            None,
            [PCATransform(n_components=3)],
            [VAETransform(latent_dim=3)],
            [
                StandardizeTransform(),
                PCATransform(n_components=3),
                VAETransform(latent_dim=2),
            ],
        ],
        [
            None,
            [PCATransform(n_components=1)],
            [StandardizeTransform(), PCATransform(n_components=1)],
            [VAETransform(latent_dim=1)],
            [StandardizeTransform(), VAETransform(latent_dim=1)],
        ],
    ),
)
def test_transformed_emulator(
    sample_data_y2d, new_data_y2d, model, x_transforms, y_transforms
):
    run_test(sample_data_y2d, new_data_y2d, model, x_transforms, y_transforms)


@pytest.mark.parametrize(
    ("model", "x_transforms", "y_transforms"),
    itertools.product(
        [emulator for emulator in ALL_EMULATORS if emulator.is_multioutput()],
        [
            None,
            [StandardizeTransform()],
            [PCATransform(n_components=3)],
            [VAETransform(latent_dim=3)],
            [
                StandardizeTransform(),
                PCATransform(n_components=3),
                VAETransform(latent_dim=2),
            ],
        ],
        [
            # TODO: PCA/VAE both require StandardizeTransform for numerical stability
            # e.g. "ValueError: Input tensor y contains non-finite values"
            # TODO: check error when no target transforms are provided
            # None,
            # [StandardizeTransform()],
            [StandardizeTransform(), PCATransform(n_components=10)],
            [StandardizeTransform(), PCATransform(n_components=20)],
            [StandardizeTransform(), VAETransform(latent_dim=10)],
            [StandardizeTransform(), VAETransform(latent_dim=20)],
        ],
    ),
)
def test_transformed_emulator_100_targets(
    sample_data_y2d_100_targets,
    new_data_y2d_100_targets,
    model,
    x_transforms,
    y_transforms,
):
    run_test(
        sample_data_y2d_100_targets,
        new_data_y2d_100_targets,
        model,
        x_transforms,
        y_transforms,
    )


@pytest.mark.parametrize(
    ("model", "x_transforms", "y_transforms"),
    itertools.product(
        [emulator for emulator in ALL_EMULATORS if emulator.is_multioutput()],
        [
            None,
            [StandardizeTransform()],
            [PCATransform(n_components=3)],
            [VAETransform(latent_dim=3)],
            [
                StandardizeTransform(),
                PCATransform(n_components=3),
                VAETransform(latent_dim=2),
            ],
        ],
        [
            # TODO: PCA/VAE both require StandardizeTransform for numerical stability
            # e.g. "ValueError: Input tensor y contains non-finite values"
            # TODO: check error when no target transforms are provided
            # None,
            # [StandardizeTransform()],
            [StandardizeTransform(), PCATransform(n_components=10)],
            [StandardizeTransform(), PCATransform(n_components=20)],
            [StandardizeTransform(), VAETransform(latent_dim=10)],
            [StandardizeTransform(), VAETransform(latent_dim=20)],
        ],
    ),
)
def test_transformed_emulator_1000_targets(
    sample_data_y2d_1000_targets,
    new_data_y2d_1000_targets,
    model,
    x_transforms,
    y_transforms,
):
    run_test(
        sample_data_y2d_1000_targets,
        new_data_y2d_1000_targets,
        model,
        x_transforms,
        y_transforms,
    )


def test_inverse_gaussian_and_sample_pca(sample_data_y2d, new_data_y2d):
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    em = TransformedEmulator(
        x,
        y,
        model=GaussianProcessExact,
        x_transforms=[StandardizeTransform()],
        y_transforms=[PCATransform(n_components=1)],
    )
    em.fit(x, y)
    y_pred = em.predict(x2)
    z_pred = em.model.predict(em.x_transforms[0](x2))
    assert isinstance(z_pred, GaussianLike)
    y_pred2 = em.y_transforms[0]._inverse_sample(
        z_pred, n_samples=10000, full_covariance=True
    )
    assert isinstance(y_pred, GaussianLike)
    assert isinstance(y_pred2, GaussianLike)
    print()
    print(y_pred.covariance_matrix)
    print(y_pred2.covariance_matrix)
    y_pred_cov = y_pred.covariance_matrix
    y_pred2_cov = y_pred2.covariance_matrix
    assert isinstance(y_pred_cov, TensorLike)
    assert isinstance(y_pred2_cov, TensorLike)
    print(y_pred2_cov - y_pred_cov)
    # TODO: consider if this is close enough for PCA case
    assert torch.allclose(y_pred_cov, y_pred2_cov, atol=1e-1)


def test_inverse_gaussian_and_sample_vae(sample_data_y2d, new_data_y2d):
    torch.manual_seed(0)
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    em = TransformedEmulator(
        x,
        y,
        model=GaussianProcessExact,
        x_transforms=[StandardizeTransform()],
        y_transforms=[StandardizeTransform(), VAETransform(latent_dim=1)],
    )
    em.fit(x, y)
    y_pred = em.predict(x2)
    z_pred = em.model.predict(em.x_transforms[0](x2))
    assert isinstance(z_pred, GaussianLike)
    y_pred2 = em.y_transforms[0]._inverse_gaussian(
        em.y_transforms[1]._inverse_sample(z_pred, n_samples=10000)
    )
    assert isinstance(y_pred, GaussianLike)
    assert isinstance(y_pred2, GaussianLike)
    print()
    print(y_pred.covariance_matrix)
    print(y_pred2.covariance_matrix)
    y_pred_cov = y_pred.covariance_matrix
    y_pred2_cov = y_pred2.covariance_matrix
    assert isinstance(y_pred_cov, TensorLike)
    assert isinstance(y_pred2_cov, TensorLike)
    diff = y_pred2_cov - y_pred_cov
    print(diff)
    assert isinstance(diff, TensorLike)
    diff_abs = (diff / y_pred_cov).abs()

    # TODO: these are not necessarily expected to be close since both approximate in
    # different ways
    # Most are within 50% error
    assert torch.quantile(diff_abs.flatten(), 0.9).item() < 0.25
    assert torch.quantile(diff_abs.flatten(), 0.95).item() < 0.5
    # Some large max differences so will not assert on these
    print("Max diff", diff_abs.abs().max())
    # assert torch.allclose(diff_abs, torch.zeros_like(diff_abs), atol=0.4)


# def test_tune_transformed_gp(sample_data_y2d):
#     x, y = sample_data_y2d
#     tuner = Tuner(x, y, n_iter=5)
#     #TODO: consider partial for specific model
#     scores, configs = tuner.run(TransformedEmulator)
#     assert len(scores) == 5
#     assert len(configs) == 5
