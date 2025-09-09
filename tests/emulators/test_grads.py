import itertools

import pytest
import torch
from autoemulate.core.types import TensorLike
from autoemulate.emulators import (
    GAUSSIAN_PROCESS_EMULATORS,
    PYTORCH_EMULATORS,
    RadialBasisFunctions,
)
from autoemulate.emulators.transformed.base import TransformedEmulator
from autoemulate.transforms.pca import PCATransform
from autoemulate.transforms.standardize import StandardizeTransform
from autoemulate.transforms.vae import VAETransform


@pytest.mark.parametrize(
    ("emulator", "x_transforms", "y_transforms"),
    itertools.product(
        [emulator for emulator in PYTORCH_EMULATORS if emulator.is_multioutput()],
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
def test_grads(sample_data_y2d, new_data_y2d, emulator, x_transforms, y_transforms):
    if emulator == RadialBasisFunctions:
        pytest.xfail("RadialBasisFunctions do not appear to support gradients")

    # Test gradient computation for the given emulator
    x, y = sample_data_y2d
    x2, _ = new_data_y2d

    # Fit emulator
    em = TransformedEmulator(
        x, y, x_transforms=x_transforms, y_transforms=y_transforms, model=emulator
    )
    em.fit(x, y)

    # Get predictions
    # Set x2 as requires grad here so gradient can be taken
    x2.requires_grad = True
    mean, variance = em.predict_mean_and_variance(x2, with_grad=True)

    # Check that mean requires gradients
    assert mean.requires_grad
    if variance is not None:
        assert variance.requires_grad

    for output in [mean, variance] if variance is not None else [mean]:
        grads = torch.autograd.grad(outputs=output.sum(), inputs=x2, retain_graph=True)[
            0
        ]
        assert grads.shape == x2.shape  # type: ignore  # noqa: PGH003

        # Check that gradients were computed for the input
        assert grads is not None
        assert not torch.allclose(grads, torch.zeros_like(grads))


@pytest.mark.parametrize(
    ("emulator", "x_transforms", "y_transforms"),
    itertools.product(
        [emulator for emulator in PYTORCH_EMULATORS if emulator.is_multioutput()],
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
def test_grads_func(
    sample_data_y2d, new_data_y2d, emulator, x_transforms, y_transforms
):
    if emulator == RadialBasisFunctions:
        pytest.xfail("RadialBasisFunctions do not appear to support gradients")
    if emulator in GAUSSIAN_PROCESS_EMULATORS:
        pytest.xfail(
            "GaussianProcess emulators do not support torch.func API for gradients as "
            "LinearOperator does not support gradients with torch.func API."
        )
    if emulator.__name__.startswith("Ensemble"):
        pytest.xfail(
            "Ensemble emulators do not support torch.func API for gradients as "
            "LinearOperator does not support gradients with torch.func API."
        )
    # Test gradient computation for the given emulator
    x, y = sample_data_y2d
    x2, _ = new_data_y2d

    # Fit emulator
    em = TransformedEmulator(
        x, y, x_transforms=x_transforms, y_transforms=y_transforms, model=emulator
    )
    em.fit(x, y)

    def sum_of_output(index: int):
        def f(x: TensorLike) -> TensorLike:
            m, v = em.predict_mean_and_variance(x, with_grad=True)
            out = m if index == 0 else v
            assert isinstance(out, TensorLike)
            return out.sum()

        return f

    for idx in [0, 1] if em.supports_uq else [0]:
        grads = torch.func.grad(sum_of_output(idx))(x2)
        assert isinstance(grads, TensorLike)
        assert grads.shape == x2.shape
        assert grads is not None
        assert not torch.allclose(grads, torch.zeros_like(grads))
