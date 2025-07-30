import itertools

import pytest
import torch
from autoemulate.experimental.core.types import (
    DistributionLike,
    GaussianLike,
    TensorLike,
)
from autoemulate.experimental.data.utils import set_random_seed
from autoemulate.experimental.emulators import ALL_EMULATORS, GaussianProcess
from autoemulate.experimental.emulators.base import ProbabilisticEmulator
from autoemulate.experimental.emulators.transformed.base import TransformedEmulator
from autoemulate.experimental.transforms import (
    PCATransform,
    StandardizeTransform,
    VAETransform,
)


def run_test(train_data, test_data, model, x_transforms, y_transforms):
    x, y = train_data
    x2, _ = test_data
    em = TransformedEmulator(
        x, y, x_transforms=x_transforms, y_transforms=y_transforms, model=model
    )
    em.fit(x, y)
    y_pred = em.predict(x2)
    if issubclass(model, ProbabilisticEmulator):
        assert isinstance(y_pred, DistributionLike)
        assert y_pred.mean.shape == (x2.shape[0], y.shape[1])
        assert not y_pred.mean.requires_grad
    else:
        assert isinstance(y_pred, TensorLike)
        assert y_pred.shape == (x2.shape[0], y.shape[1])
        assert not y_pred.requires_grad


@pytest.mark.parametrize(
    ("model", "x_transforms", "y_transforms"),
    itertools.product(
        [emulator for emulator in ALL_EMULATORS if emulator.is_multioutput()],
        [
            None,
            [StandardizeTransform(), PCATransform(n_components=3)],
            [StandardizeTransform(), VAETransform(latent_dim=3)],
        ],
        [
            None,
            [StandardizeTransform()],
            [StandardizeTransform(), PCATransform(n_components=1)],
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
        [emulator for emulator in ALL_EMULATORS if emulator.supports_grad],
        [
            None,
            [StandardizeTransform()],
        ],
        [
            None,
            [StandardizeTransform()],
        ],
    ),
)
def test_transformed_emulator_grad(
    sample_data_y2d, new_data_y2d, model, x_transforms, y_transforms
):
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    em = TransformedEmulator(
        x, y, x_transforms=x_transforms, y_transforms=y_transforms, model=model
    )
    em.fit(x, y)
    y_pred = em.predict(x2)
    if issubclass(model, ProbabilisticEmulator):
        assert isinstance(y_pred, DistributionLike)
        assert not y_pred.mean.requires_grad
    else:
        assert isinstance(y_pred, TensorLike)
        assert not y_pred.requires_grad

    y_pred_grad = em.predict(x2, with_grad=True)
    if issubclass(model, ProbabilisticEmulator):
        assert isinstance(y_pred_grad, DistributionLike)
        assert y_pred_grad.mean.requires_grad
    else:
        assert isinstance(y_pred_grad, TensorLike)
        assert y_pred_grad.requires_grad


@pytest.mark.parametrize(
    ("model", "x_transforms"),
    itertools.product(
        [
            emulator
            for emulator in ALL_EMULATORS
            if emulator.supports_grad and emulator.supports_grad
        ],
        [[PCATransform(n_components=3)], [VAETransform(latent_dim=3)]],
    ),
)
def test_transformed_emulator_no_grad(
    sample_data_y2d, new_data_y2d, model, x_transforms
):
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    em = TransformedEmulator(
        x, y, x_transforms=x_transforms, y_transforms=None, model=model
    )
    em.fit(x, y)
    y_pred = em.predict(x2)
    if issubclass(model, ProbabilisticEmulator):
        assert isinstance(y_pred, DistributionLike)
        assert not y_pred.mean.requires_grad
    else:
        assert isinstance(y_pred, TensorLike)
        assert not y_pred.requires_grad

    with pytest.raises(ValueError, match="Gradient calculation is not supported."):
        em.predict(x2, with_grad=True)


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
            # TODO: revisit failing case with largr number of targets and no transforms
            # None,
            [StandardizeTransform()],
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
            # TODO: revisit failing case with largr number of targets and no transforms
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
    set_random_seed(0)
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    em = TransformedEmulator(
        x,
        y,
        model=GaussianProcess,
        x_transforms=[StandardizeTransform()],
        y_transforms=[StandardizeTransform(), PCATransform(n_components=1)],
    )
    em.fit(x, y)

    # Get predicted distribution from the emulator
    y_pred = em.predict(x2)

    # Get predicted latent and reconstruct through sampling
    z_pred = em.model.predict(em.x_transforms[0](x2), with_grad=False)
    assert isinstance(z_pred, GaussianLike)

    # Test inverse sampling through only PCA
    y_pred2 = em.y_transforms[0]._inverse_gaussian(
        em.y_transforms[1]._inverse_sample(z_pred, n_samples=10000)
    )
    assert isinstance(y_pred, GaussianLike)
    assert isinstance(y_pred2, GaussianLike)
    y_pred_cov = y_pred.covariance_matrix
    y_pred2_cov = y_pred2.covariance_matrix
    assert isinstance(y_pred_cov, TensorLike)
    assert isinstance(y_pred2_cov, TensorLike)

    print((y_pred2_cov - y_pred_cov).abs().max())
    print(((y_pred2_cov - y_pred_cov).abs() / y_pred_cov.abs()).max())
    assert torch.allclose(y_pred_cov, y_pred2_cov, rtol=5e-2)


def test_inverse_gaussian_and_sample_vae(sample_data_y2d, new_data_y2d):
    torch.manual_seed(0)
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    em = TransformedEmulator(
        x,
        y,
        model=GaussianProcess,
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

    print("Max diff", diff_abs.abs().max())
    print((y_pred2_cov - y_pred_cov).abs().max())
    print(((y_pred2_cov - y_pred_cov).abs() / y_pred_cov.abs()).max())

    # Assert with around 40% error to account for the stochastic nature of VAE sampling
    assert torch.allclose(y_pred2_cov, y_pred_cov, rtol=0.4)
