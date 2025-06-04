import itertools

import pytest
from autoemulate.experimental.emulators import ALL_EMULATORS, GaussianProcessExact
from autoemulate.experimental.emulators.transformed.base import TransformedEmulator
from autoemulate.experimental.transforms import (
    PCATransform,
    StandardizeTransform,
    VAETransform,
)

# from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import DistributionLike, TensorLike


@pytest.mark.parametrize(
    ("model", "transform", "target_transforms"),
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
            [VAETransform(latent_dim=1)],
            [StandardizeTransform(), VAETransform(latent_dim=1)],
        ],
    ),
)
def test_transformed_emulator(
    sample_data_y2d, new_data_y2d, model, transform, target_transforms
):
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    em = TransformedEmulator(
        x,
        y,
        transforms=transform,
        target_transforms=target_transforms,
        model=model,
    )
    em.fit(x, y)
    y_pred = em.predict(x2)
    if model is GaussianProcessExact:
        assert isinstance(y_pred, DistributionLike)
        assert y_pred.mean.shape == (20, 2)
    else:
        assert isinstance(y_pred, TensorLike)
        assert y_pred.shape == (20, 2)


# def test_tune_transformed_gp(sample_data_y2d):
#     x, y = sample_data_y2d
#     tuner = Tuner(x, y, n_iter=5)
#     #TODO: consider partial for specific model
#     scores, configs = tuner.run(TransformedEmulator)
#     assert len(scores) == 5
#     assert len(configs) == 5
