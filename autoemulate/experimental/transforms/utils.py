import torch


# TODO: consider if this function is sufficiently robust for all cases.
# TODO: compare with linear_operator's psd_safe_cholesky:
# `from linear_operator.utils.cholesky import psd_safe_cholesky``
def make_positive_definite(
    cov, epsilon=1e-6, min_eigval=1e-6, max_tries_epsilon=3, max_tries_min_eigval=5
):
    # Attempt with epsilon first
    for _ in range(max_tries_epsilon):
        try:
            torch.linalg.cholesky(cov)
            return cov
        except RuntimeError:
            cov = cov + epsilon * torch.eye(
                cov.shape[0], device=cov.device, dtype=cov.dtype
            )
            # Ensure symmetry
            cov = (cov + cov.T) / 2
            epsilon *= 10

    # Spectral approach by clamping eigenvalues
    for _ in range(max_tries_min_eigval):
        try:
            torch.linalg.cholesky(cov)
            return cov
        except RuntimeError:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals = torch.clamp(eigvals, min=min_eigval)
            cov = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            # Ensure symmetry
            cov = (cov + cov.T) / 2
            min_eigval *= 10
    msg = f"Matrix could not be made positive definite:\n{cov}"
    raise RuntimeError(msg)
