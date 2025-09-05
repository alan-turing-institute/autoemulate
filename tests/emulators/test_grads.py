import pytest
import torch
from autoemulate.core.types import TensorLike
from autoemulate.emulators import (
    GAUSSIAN_PROCESS_EMULATORS,
    PYTORCH_EMULATORS,
    RadialBasisFunctions,
)
from autoemulate.emulators.base import Emulator


@pytest.mark.parametrize(
    "emulator",
    # RadialBasisFunctions do not appear to support gradients
    [emulator for emulator in PYTORCH_EMULATORS if emulator != RadialBasisFunctions],
)
def test_grads(sample_data_y1d, new_data_y1d, emulator):
    # Test gradient computation for the given emulator
    x, y = sample_data_y1d
    x2, _ = new_data_y1d

    # Fit emulator
    em: Emulator = emulator(x, y)
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
    "emulator",
    [
        emulator
        for emulator in PYTORCH_EMULATORS
        # RadialBasisFunctions do not appear to support gradients
        if emulator != RadialBasisFunctions
        # GaussianProcess emulators do not appear to support torch.func API for grads
        and emulator not in GAUSSIAN_PROCESS_EMULATORS
    ],
)
def test_grads_func(sample_data_y1d, new_data_y1d, emulator):
    # Test gradient computation for the given emulator
    x, y = sample_data_y1d
    x2, _ = new_data_y1d

    # Fit emulator
    em: Emulator = emulator(x, y)
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
