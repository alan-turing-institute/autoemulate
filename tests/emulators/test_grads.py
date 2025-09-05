import pytest
import torch
from autoemulate.emulators import PYTORCH_EMULATORS, RadialBasisFunctions
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
