import pytest


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_make_positive_definite(scale):
    """Test that make_positive_definite can handle various cases of cov matrices."""
    import torch
    from autoemulate.transforms.utils import make_positive_definite

    # Try large scale
    for dim in [5, 20, 100]:
        # Batch of 1000 random covariance matrices
        covs = torch.rand(1000, dim, dim) * scale
        covs = make_positive_definite(covs, clamp_eigvals=True)

        for cov in covs:
            assert cov.shape == (dim, dim)
            # Try cholesky decomposition to check if it's positive definite
            torch.linalg.cholesky(cov)
