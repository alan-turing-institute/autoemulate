import torch


def make_positive_definite(cov, epsilon=1e-6, max_tries=3):
    for _ in range(max_tries):
        try:
            torch.linalg.cholesky(cov)
            return cov
        except RuntimeError:
            cov = cov + epsilon * torch.eye(
                cov.shape[0], device=cov.device, dtype=cov.dtype
            )
            epsilon *= 10
    msg = "Matrix could not be made positive definite."
    raise RuntimeError(msg)
