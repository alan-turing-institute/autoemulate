import pytest
from autoemulate.experimental.emulators import GaussianProcessExact
from autoemulate.experimental.transforms import PCATransform, VAETransform


@pytest.mark.parametrize(
    ("transform"), [PCATransform(n_components=2), VAETransform(latent_dim=2)]
)
def test_transform_shapes(sample_data_y2d, transform):
    x, _ = sample_data_y2d
    transform.fit(x)
    z = transform(x)
    assert z.shape == (20, 2)
    assert transform.inv(z).shape == (20, 5)


@pytest.mark.parametrize(
    ("transform"), [PCATransform(n_components=2), VAETransform(latent_dim=2)]
)
def test_transform_inverse_for_gaussians(sample_data_y2d, transform):
    x, y = sample_data_y2d
    transform.fit(y)
    z = transform(y)
    gp = GaussianProcessExact(x, z)
    z_pred = gp.predict(x[: x.shape[0] // 2])
    for method in [transform._inverse_sample, transform._inverse_gaussian]:
        y_pred = method(z_pred)
        assert y_pred.mean.shape == (10, 2)
