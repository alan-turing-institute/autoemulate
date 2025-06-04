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


def run_test(train_data, test_data, model, transform, target_transforms):
    x, y = train_data
    x2, _ = test_data
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
        assert y_pred.mean.shape == (x2.shape[0], y.shape[1])
    else:
        assert isinstance(y_pred, TensorLike)
        assert y_pred.shape == (x2.shape[0], y.shape[1])


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
    run_test(sample_data_y2d, new_data_y2d, model, transform, target_transforms)


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
            # TODO: check error when no target transforms are provided
            # None,
            [PCATransform(n_components=10)],
            [PCATransform(n_components=10)],
            [PCATransform(n_components=15)],
            # TODO: check error for n_components = 20
            # ValueError: Input tensor y contains non-finite values
            # [PCATransform(n_components=20)],
            [VAETransform(latent_dim=10)],
            # TODO: check error for latent_dim = 20
            # ValueError: Input tensor y contains non-finite values
            # [VAETransform(latent_dim=20)],
            [StandardizeTransform()],
            [StandardizeTransform(), VAETransform(latent_dim=10)],
        ],
    ),
)
def test_transformed_emulator_100_targets(
    sample_data_y2d_100_targets,
    new_data_y2d_100_targets,
    model,
    transform,
    target_transforms,
):
    run_test(
        sample_data_y2d_100_targets,
        new_data_y2d_100_targets,
        model,
        transform,
        target_transforms,
    )


# def test_tune_transformed_gp(sample_data_y2d):
#     x, y = sample_data_y2d
#     tuner = Tuner(x, y, n_iter=5)
#     #TODO: consider partial for specific model
#     scores, configs = tuner.run(TransformedEmulator)
#     assert len(scores) == 5
#     assert len(configs) == 5
