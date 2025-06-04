import torch


def make_positive_definite(cov, epsilon=1e-6, min_eigval=1e-6, max_tries=4):
    # Attempt with epsilon first
    for _ in range(max_tries):
        try:
            torch.linalg.cholesky(cov)
            return cov
        except RuntimeError:
            cov = cov + epsilon * torch.eye(
                cov.shape[0], device=cov.device, dtype=cov.dtype
            )
            # Symmetrize?
            # cov = (cov + cov.T) / 2
            epsilon *= 10

    # Spectral approach by clamping eigenvalues
    min_eigval = 1e-6
    for _ in range(max_tries):
        try:
            torch.linalg.cholesky(cov)
            return cov
        except RuntimeError:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals = torch.clamp(eigvals, min=min_eigval)
            cov = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            min_eigval *= 10
    msg = f"Matrix could not be made positive definite:\n{cov}"
    raise RuntimeError(msg)
