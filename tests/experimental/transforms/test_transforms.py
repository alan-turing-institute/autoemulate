import pytest
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
def test_transform_sample(sample_data_y2d, transform): ...
