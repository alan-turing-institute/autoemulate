import numpy as np
import pytest
import torch
from autoemulate.core.types import GaussianLike, TensorLike
from autoemulate.emulators import GaussianProcess
from autoemulate.transforms import (
    PCATransform,
    StandardizeTransform,
    VAETransform,
)
from autoemulate.transforms.base import (
    AutoEmulateTransform,
    _inverse_sample_gaussian_like,
    _is_affine_empirical,
    is_affine,
)
from sklearn.decomposition import PCA as SklearnPCA
from torch.distributions.transforms import ReshapeTransform


@pytest.mark.parametrize(
    ("transform", "expected_shape"),
    [
        (PCATransform(n_components=2), (20, 2)),
        (VAETransform(latent_dim=2), (20, 2)),
        (StandardizeTransform(), (20, 5)),
    ],
)
def test_transform_shapes(sample_data_y2d, transform, expected_shape):
    x, _ = sample_data_y2d
    transform.fit(x)
    z = transform(x)
    assert z.shape == expected_shape
    assert transform.inv(z).shape == (20, 5)


@pytest.mark.parametrize(
    ("transform", "affine"),
    [
        (PCATransform(n_components=2), True),
        (VAETransform(latent_dim=2), False),
        (StandardizeTransform(), True),
        # Only reshape event dims (exclude batch): 5 -> (5, 1)
        (ReshapeTransform((5,), (5, 1)), True),
    ],
)
def test_is_transform_affine(sample_data_y2d, transform, affine):
    x, _ = sample_data_y2d
    if isinstance(transform, AutoEmulateTransform):
        transform.fit(x)
        assert transform.affine == affine
        assert is_affine(transform, x) == transform.affine
    assert _is_affine_empirical(transform, x) == affine


@pytest.mark.parametrize(
    ("transform"),
    [PCATransform(n_components=2), VAETransform(latent_dim=2), StandardizeTransform()],
)
def test_transform_inverse_for_gaussians(sample_data_y2d, transform):
    x, y = sample_data_y2d
    transform.fit(y)
    z = transform(y)
    gp = GaussianProcess(x, z)
    gp.fit(x, z)
    z_pred = gp.predict(x[: x.shape[0] // 2])
    for method in [transform._inverse_sample, transform._inverse_gaussian]:
        y_pred = method(z_pred)
        assert y_pred.mean.shape == (10, 2)


def test_standardize(sample_data_y2d):
    x, _ = sample_data_y2d
    std_transform = StandardizeTransform()
    std_transform.fit(x)

    # Test forward method
    z = std_transform(x)
    assert z.shape == (20, 5)
    assert torch.allclose(z.mean(dim=0), torch.zeros(5), atol=1e-6)
    assert torch.allclose(z.std(dim=0), torch.ones(5), atol=1e-6)

    # Test inverse method
    x_inv = std_transform.inv(z)
    assert isinstance(x_inv, torch.Tensor)
    assert x_inv.shape == (20, 5)
    assert torch.allclose(x_inv.mean(dim=0), x.mean(dim=0), atol=1e-6)
    assert torch.allclose(x_inv.std(dim=0), x.std(dim=0), atol=1e-6)


def test_pca(sample_data_y2d):
    pca = PCATransform(n_components=2)
    x, _ = sample_data_y2d
    pca.fit(x)
    skpca = SklearnPCA(n_components=2)
    skpca.fit(x)
    assert np.allclose(
        np.abs(pca.components.cpu().numpy()), np.abs(skpca.components_.T), atol=1e-6
    )

    # Test forward method
    z = pca(x)
    z_sk = skpca.transform(x)
    assert isinstance(z, torch.Tensor)
    assert z.shape == (20, 2)
    assert np.allclose(np.abs(z.cpu().numpy()), np.abs(z_sk), atol=1e-6)

    # Test inverse method
    x_inv = pca.inv(z)
    x_inv_sk = skpca.inverse_transform(z_sk)
    assert isinstance(x_inv, torch.Tensor)
    assert x_inv.shape == (20, 5)
    print(x_inv)
    print(x_inv_sk)
    assert np.allclose(x_inv.cpu().numpy(), x_inv_sk, atol=1e-6)


@pytest.mark.parametrize(
    ("loc", "scale", "expected_shape"),
    [
        (torch.zeros(4), torch.eye(4), (10, 10)),
        (torch.zeros(1, 4), torch.eye(4).repeat(1, 1, 1), (1, 10, 10)),
        (torch.zeros(3, 4), torch.eye(4).repeat(3, 1, 1), (3, 10, 10)),
    ],
)
def test_inverse_sample_gaussian_like(loc, scale, expected_shape):
    n = 100
    y_t = GaussianLike(loc, scale)
    pca = PCATransform(n_components=4)
    pca.fit(torch.randn(100, 10))

    y = _inverse_sample_gaussian_like(pca.inv, y_t, n_samples=n, full_covariance=False)
    assert isinstance(y.covariance_matrix, TensorLike)
    assert y.covariance_matrix.shape == expected_shape

    y = _inverse_sample_gaussian_like(pca.inv, y_t, n_samples=n, full_covariance=True)
    assert isinstance(y.covariance_matrix, TensorLike)
    assert y.covariance_matrix.shape == expected_shape
