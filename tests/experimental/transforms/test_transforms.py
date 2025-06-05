import numpy as np
import pytest
from autoemulate.experimental.emulators import GaussianProcessExact
from autoemulate.experimental.transforms import (
    PCATransform,
    StandardizeTransform,
    VAETransform,
)
from sklearn.decomposition import PCA as SklearnPCA


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
    ("transform"),
    [PCATransform(n_components=2), VAETransform(latent_dim=2), StandardizeTransform()],
)
def test_transform_inverse_for_gaussians(sample_data_y2d, transform):
    x, y = sample_data_y2d
    transform.fit(y)
    z = transform(y)
    gp = GaussianProcessExact(x, z)
    gp.fit(x, z)
    z_pred = gp.predict(x[: x.shape[0] // 2])
    for method in [transform._inverse_sample, transform._inverse_gaussian]:
        y_pred = method(z_pred)
        assert y_pred.mean.shape == (10, 2)


def test_pca(sample_data_y2d):
    pca = PCATransform(n_components=2, niter=50)
    x, _ = sample_data_y2d
    pca.fit(x)
    skpca = SklearnPCA(n_components=2)
    skpca.fit(x)
    assert np.allclose(
        np.abs(pca.components.cpu().numpy()), np.abs(skpca.components_.T), atol=1e-6
    )
