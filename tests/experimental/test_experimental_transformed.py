from autoemulate.experimental.emulators.gaussian_process.exact import (
    GaussianProcessExact,
)
from autoemulate.experimental.emulators.transformed.base import TransformedEmulator
from autoemulate.experimental.transforms.pca import PCATransform

# from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import DistributionLike


def test_transformed_gp(sample_data_y2d, new_data_y2d):
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    em = TransformedEmulator(
        x,
        y,
        transforms=[PCATransform(n_components=1)],
        target_transforms=[PCATransform(n_components=1)],
        model=GaussianProcessExact,
    )

    em.fit(x, y)
    y_pred = em.predict(x2)
    assert isinstance(y_pred, DistributionLike)
    assert y_pred.mean.shape == (20, 2)


# def test_tune_transformed_gp(sample_data_y2d):
#     x, y = sample_data_y2d
#     tuner = Tuner(x, y, n_iter=5)
#     #TODO: consider partial for specific model
#     scores, configs = tuner.run(TransformedEmulator)
#     assert len(scores) == 5
#     assert len(configs) == 5
