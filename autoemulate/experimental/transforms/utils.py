import warnings

import torch
from linear_operator.utils.warnings import NumericalWarning


def make_positive_definite(
    cov, epsilon=1e-6, min_eigval=1e-6, max_tries_epsilon=3, max_tries_min_eigval=6
):
    """Ensure a covariance matrix is positive definite by:
        1. adding `epsilon` to the diagonal and symmetrizing the matrix
        2. if this fails, clamping eigenvalues to a minimum value `min_eigval`

    See related function in linear_operator: `psd_safe_cholesky`.

    Parameters
    ----------
    cov (torch.Tensor): The covariance matrix to be made positive definite.
    epsilon (float): Initial value to add to the diagonal for numerical stability.
        Default is 1e-6.
    min_eigval (float): Minimum eigenvalue to clamp to if the matrix is not positive
        definite.
        Default is 1e-6.
    max_tries_epsilon (int): Maximum number of attempts to add `epsilon` to the
        diagonal.
        Default is 3.
    max_tries_min_eigval (int): Maximum number of attempts to clamp eigenvalues.
        Default is 1.

    Returns
    -------
    torch.Tensor: A positive definite covariance matrix.

    """
    # Attempt with epsilon first
    for i in range(max_tries_epsilon):
        try:
            torch.linalg.cholesky(cov)
            if i > 0:
                warnings.warn(
                    f"cov not p.d. - added {epsilon:.1e} to the diagonal and "
                    "symmetrized",
                    NumericalWarning,
                    stacklevel=2,
                )
            return cov
        except RuntimeError:
            eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
            if len(cov.shape) == 3:
                eye = eye.unsqueeze(0)
            if len(cov.shape) > 3:
                msg = f"cov must be a 2D or 3D tensor, got shape {cov.shape}"
                raise ValueError(msg) from None
            cov = cov + epsilon * eye
            # Ensure symmetry
            cov = (cov + cov.T) / 2
            epsilon *= 10

    # Spectral approach by clamping eigenvalues
    for i in range(max_tries_min_eigval):
        try:
            torch.linalg.cholesky(cov)
            if i > 0:
                warnings.warn(
                    f"cov not p.d. - clamped eigval to {min_eigval:.1e} and "
                    "symmetrized",
                    NumericalWarning,
                    stacklevel=2,
                )
            return cov
        except RuntimeError as e:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals = torch.clamp(eigvals, min=min_eigval)
            cov = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            # Ensure symmetry
            cov = (cov + cov.T) / 2
            min_eigval *= 10
            if i == max_tries_min_eigval - 1:
                msg = f"Matrix could not be made positive definite:\n{cov}"
                raise RuntimeError(msg) from e
    return None
