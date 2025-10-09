import itertools

import pytest
import torch
from autoemulate.core.types import DistributionLike, GaussianLike, TensorLike
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators import DEFAULT_EMULATORS
from autoemulate.emulators.base import ProbabilisticEmulator
from autoemulate.emulators.gaussian_process.exact import GaussianProcess
from autoemulate.emulators.transformed.base import TransformedEmulator
from autoemulate.transforms import PCATransform, StandardizeTransform, VAETransform
from autoemulate.transforms.base import AutoEmulateTransform


def get_pytest_param_yof(model, x_t, y_t, o, f):
    return (
        pytest.param(
            model,
            x_t,
            y_t,
            o,
            f,
            marks=pytest.mark.xfail(
                raises=NotImplementedError,
                reason="Full covariance sampling not implemented",
            ),
        )
        if (o and f and model.supports_uq)
        else (model, x_t, y_t, o, f)
    )


def get_pytest_param_of(model, x_t, o, f):
    return (
        pytest.param(
            model,
            x_t,
            o,
            f,
            marks=pytest.mark.xfail(
                raises=NotImplementedError,
                reason="Full covariance sampling not implemented",
            ),
        )
        if (o and f and model.supports_uq)
        else (model, x_t, o, f)
    )


def get_pytest_param_yo(model, x_t, y_t, o):
    return pytest.param(model, x_t, y_t, o)


def run_test(
    train_data,
    test_data,
    model,
    x_transforms,
    y_transforms,
    output_from_samples,
    full_covariance,
    test_grads=True,
):
    if not model.is_multioutput() and (
        y_transforms is None
        or (
            y_transforms is not None
            and not isinstance(y_transforms[-1], PCATransform | VAETransform)
        )
    ):
        pytest.skip("Only multioutput models supported for this test case.")
    x, y = train_data
    x2, _ = test_data
    em = TransformedEmulator(
        x,
        y,
        x_transforms=x_transforms,
        y_transforms=y_transforms,
        model=model,
        output_from_samples=output_from_samples,
        full_covariance=full_covariance,
    )
    em.fit(x, y)
    y_pred = em.predict(x2)
    if issubclass(model, ProbabilisticEmulator):
        assert isinstance(y_pred, DistributionLike)
        y_pred_mean = em.predict_mean(x2)
        assert y_pred_mean.shape == (x2.shape[0], y.shape[1])
        assert not y_pred_mean.requires_grad
        y_pred_mean, y_pred_var = em.predict_mean_and_variance(x2)
        assert isinstance(y_pred_var, TensorLike)
        assert y_pred_mean.shape == (x2.shape[0], y.shape[1])
        assert y_pred_var.shape == (x2.shape[0], y.shape[1])
        assert not y_pred_mean.requires_grad
        assert not y_pred_var.requires_grad
    else:
        assert isinstance(y_pred, TensorLike)
        assert y_pred.shape == (x2.shape[0], y.shape[1])
        assert not y_pred.requires_grad

    if not test_grads:
        return

    # Test gradient support only for few targets case
    if em.supports_grad:
        y_pred_grad = em.predict(x2, with_grad=True)
        if issubclass(model, ProbabilisticEmulator):
            assert isinstance(y_pred_grad, DistributionLike)
            y_pred_grad_mean = em.predict_mean(x2, with_grad=True)
            assert y_pred_grad_mean.requires_grad
            y_pred_grad_mean, y_pred_grad_var = em.predict_mean_and_variance(
                x2, with_grad=True
            )
            assert isinstance(y_pred_grad_var, TensorLike)
            assert y_pred_grad_mean.requires_grad
            assert y_pred_grad_var.requires_grad
        else:
            assert isinstance(y_pred_grad, TensorLike)
            assert y_pred_grad.requires_grad
            y_pred_grad_mean = em.predict_mean(x2, with_grad=True)
            assert y_pred_grad_mean.requires_grad
        return

    # Test that gradients raise error when not supported
    with pytest.raises(ValueError, match="Gradient calculation is not supported."):
        em.predict(x2, with_grad=True)


@pytest.mark.parametrize(
    ("model", "x_transforms", "y_transforms", "output_from_samples", "full_covariance"),
    [
        get_pytest_param_yof(model, x_t, y_t, o, f)
        for model, x_t, y_t, o, f in itertools.product(
            DEFAULT_EMULATORS,
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
            [False, True],
            [False, True],
        )
    ],
)
def test_transformed_emulator(
    sample_data_y2d,
    new_data_y2d,
    model,
    x_transforms,
    y_transforms,
    output_from_samples,
    full_covariance,
):
    run_test(
        sample_data_y2d,
        new_data_y2d,
        model,
        x_transforms,
        y_transforms,
        output_from_samples,
        full_covariance,
        test_grads=True,
    )


@pytest.mark.parametrize(
    ("model", "x_transforms", "y_transforms", "output_from_samples"),
    [
        get_pytest_param_yo(model, x_t, y_t, o)
        for model, x_t, y_t, o in itertools.product(
            [emulator for emulator in DEFAULT_EMULATORS if emulator.is_multioutput()],
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
                [StandardizeTransform()],
                [StandardizeTransform(), PCATransform(n_components=10)],
                [StandardizeTransform(), PCATransform(n_components=20)],
                [StandardizeTransform(), VAETransform(latent_dim=10)],
                [StandardizeTransform(), VAETransform(latent_dim=20)],
            ],
            [False, True],
        )
    ],
)
def test_transformed_emulator_100_targets(
    sample_data_y2d_100_targets,
    new_data_y2d_100_targets,
    model,
    x_transforms,
    y_transforms,
    output_from_samples,
):
    run_test(
        sample_data_y2d_100_targets,
        new_data_y2d_100_targets,
        model,
        x_transforms,
        y_transforms,
        output_from_samples,
        full_covariance=False,
        test_grads=False,
    )


@pytest.mark.parametrize(
    ("model", "x_transforms", "y_transforms", "output_from_samples"),
    [
        get_pytest_param_yo(model, x_t, y_t, o)
        for model, x_t, y_t, o in itertools.product(
            [emulator for emulator in DEFAULT_EMULATORS if emulator.is_multioutput()],
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
                [StandardizeTransform(), PCATransform(n_components=10)],
                [StandardizeTransform(), PCATransform(n_components=20)],
                [StandardizeTransform(), VAETransform(latent_dim=10)],
                [StandardizeTransform(), VAETransform(latent_dim=20)],
            ],
            [False, True],
        )
    ],
)
def test_transformed_emulator_1000_targets(
    sample_data_y2d_1000_targets,
    new_data_y2d_1000_targets,
    model,
    x_transforms,
    y_transforms,
    output_from_samples,
):
    run_test(
        sample_data_y2d_1000_targets,
        new_data_y2d_1000_targets,
        model,
        x_transforms,
        y_transforms,
        output_from_samples,
        full_covariance=False,
        test_grads=False,
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
        full_covariance=True,
        output_from_samples=False,
    )
    em.fit(x, y)

    # Get predicted distribution from the emulator
    y_pred = em.predict(x2)

    # Get predicted latent and reconstruct through sampling
    x2_t = em.x_transforms[0](x2)
    assert isinstance(x2_t, TensorLike)
    z_pred = em.model.predict(x2_t, with_grad=False)
    assert isinstance(z_pred, GaussianLike)

    # Test inverse sampling through only PCA
    assert isinstance(em.y_transforms[0], AutoEmulateTransform)
    assert isinstance(em.y_transforms[1], AutoEmulateTransform)
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
        full_covariance=True,
        output_from_samples=False,
    )
    em.fit(x, y)
    y_pred = em.predict(x2)
    assert isinstance(em.x_transforms[0], AutoEmulateTransform)
    z_pred = em.model.predict(em.x_transforms[0](x2))
    assert isinstance(z_pred, GaussianLike)
    assert isinstance(em.y_transforms[0], AutoEmulateTransform)
    assert isinstance(em.y_transforms[1], AutoEmulateTransform)
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
    assert isinstance(diff, TensorLike)
    diff_abs = (diff / y_pred_cov).abs()

    print("Max diff", diff_abs.abs().max())
    print("Abs diff", (y_pred2_cov - y_pred_cov).abs().max())
    print("Relative diff", ((y_pred2_cov - y_pred_cov).abs() / y_pred_cov.abs()).max())

    # Robust relative-matrix checks
    eps = 1e-12

    # Relative Frobenius error of covariance matrices
    frob_num = torch.linalg.norm(y_pred2_cov - y_pred_cov, dim=(-2, -1))
    frob_den = torch.linalg.norm(y_pred_cov, dim=(-2, -1)).clamp_min(eps)
    rel_frob = (frob_num / frob_den).amax()

    # Relative error on correlation matrices to discount scale
    def corr_from_cov(C: TensorLike) -> TensorLike:
        d = torch.diagonal(C, dim1=-2, dim2=-1).clamp_min(eps).sqrt()
        scale = d.unsqueeze(-1) * d.unsqueeze(-2)
        return C / scale

    corr1 = corr_from_cov(y_pred_cov)
    corr2 = corr_from_cov(y_pred2_cov)
    corr_num = torch.linalg.norm(corr2 - corr1, dim=(-2, -1))
    corr_den = torch.linalg.norm(corr1, dim=(-2, -1)).clamp_min(eps)
    rel_corr = (corr_num / corr_den).amax()

    # Diagonal variance ratios should be within a reasonable factor
    var1 = torch.diagonal(y_pred_cov, dim1=-2, dim2=-1).clamp_min(eps)
    var2 = torch.diagonal(y_pred2_cov, dim1=-2, dim2=-1).clamp_min(eps)
    ratio = var2 / var1
    median_ratio = ratio.median()

    print("Relative Frobenius norm:", rel_frob)
    print("Relative correlation norm:", rel_corr)
    print("Median variance ratio:", median_ratio)

    # Checks are relatively loose due to stochasticity of VAE sampling
    assert rel_frob.item() < 0.50
    assert rel_corr.item() < 0.05
    assert 0.95 < median_ratio.item() < 1.05
