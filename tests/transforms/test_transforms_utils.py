import warnings

import pytest
import torch
from autoemulate.transforms.utils import make_positive_definite
from linear_operator.utils.warnings import NumericalWarning


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_make_positive_definite(scale):
    """make_positive_definite handles a range of badly-scaled, non-p.d. matrices."""
    for dim in [5, 20, 100]:
        # Batch of 1000 random (non-symmetric, non-p.d.) matrices
        covs = torch.rand(1000, dim, dim) * scale
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumericalWarning)
            covs = make_positive_definite(covs, clamp_eigvals=True)

        for cov in covs:
            assert cov.shape == (dim, dim)
            # Cholesky decomposition succeeds iff the matrix is positive definite
            torch.linalg.cholesky(cov)


def test_make_positive_definite_already_pd_is_unchanged():
    """An already-p.d. matrix is returned untouched and without warnings."""
    cov = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = make_positive_definite(cov)
    assert out is cov


def test_make_positive_definite_baseline_jitter_is_silent():
    """Rounding-level non-p.d.-ness is repaired by the baseline jitter, silently."""
    cov = torch.eye(4, dtype=torch.float64)
    cov[0, 0] = -1e-10  # not p.d., but well within the baseline jitter
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = make_positive_definite(cov)
    torch.linalg.cholesky(out)


def test_make_positive_definite_material_jitter_warns_and_keeps_gradients():
    """A material jitter warns; the result stays differentiable w.r.t. the input."""
    x = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    # Diagonal with one (small) negative eigenvalue; the baseline jitter is too
    # small to fix it, so the jitter has to escalate past it.
    cov = torch.diag(torch.stack([x, -1e-5 * x]))
    with pytest.warns(NumericalWarning, match="added"):
        out = make_positive_definite(cov, max_tries=8)
    torch.linalg.cholesky(out)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad)


def test_make_positive_definite_eigval_clamp_warns():
    """The eigenvalue-clamping fallback warns."""
    cov = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    with pytest.warns(NumericalWarning, match="gradients"):
        out = make_positive_definite(cov, clamp_eigvals=True)
    torch.linalg.cholesky(out)


def test_make_positive_definite_raises_without_clamp():
    """Without eigenvalue clamping, an unrepairable matrix raises rather than warns."""
    cov = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    with pytest.raises(RuntimeError):
        make_positive_definite(cov)


def test_make_positive_definite_rejects_high_rank_tensors():
    """Only 2D and 3D (batched) tensors are accepted."""
    with pytest.raises(ValueError, match="2D or 3D"):
        make_positive_definite(torch.zeros(2, 3, 3, 3))
