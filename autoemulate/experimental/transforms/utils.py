import torch


def make_positive_definite(cov, epsilon=1e-6, max_tries=3):
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

    # Spectral approach max eigenvalue
    epsilon = 1e-6
    for _ in range(max_tries):
        try:
            torch.linalg.cholesky(cov)
            return cov
        except RuntimeError:
            eigvals, eigvecs = torch.linalg.eigh(
                cov, device=cov.device, dtype=cov.dtype
            )
            eigvals = torch.clamp(eigvals, min=epsilon)
            cov = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            epsilon *= 10
    msg = "Matrix could not be made positive definite."
    raise RuntimeError(msg)
