import warnings

import torch
from linear_operator.utils.warnings import NumericalWarning


def make_positive_definite(
    cov, min_jitter: float = 1e-6, max_tries: int = 3, clamp_eigvals: bool = False
):
    """
    Ensure a covariance matrix is positive definite.

    Ensure a covariance matrix is positive definite by:
        1. symmetrizing the matrix
        2. adding increasing amounts of jitter to the diagonal
        3. if this fails, clamping eigenvalues to a minimum value
        4. if this still fails, clamping eigenvalues to both minimum and maximum
           values based on the median eigenvalue

    The baseline jitter repair is silent because it is expected for numerically
    semidefinite covariance estimates such as zero ensemble variance. Larger
    repairs and eigenvalue clamping still warn.

    See related function in linear_operator: `psd_safe_cholesky`:
    https://github.com/cornellius-gp/linear_operator/blob/dca438e47dd8a380d0f4e6b30c406e187062c8bd/linear_operator/utils/cholesky.py#L12

    Parameters
    ----------
    cov: torch.Tensor
        The covariance matrix to be made positive definite. Must be a 2D or 3D tensor.
    min_jitter: float
        Starting value for jitter to add to the diagonal. Jitter is increased by
        powers of 10 for each attempt. Defaults to 1e-6.
    max_tries: int
        Number of attempts to add jitter before moving to eigenvalue clamping
        (if enabled) or raising an error. Defaults to 3.
    clamp_eigvals: bool
        Whether to attempt eigenvalue clamping if adding jitter fails. If False,
        will raise an error after max_tries jitter attempts. Defaults to False.

    Returns
    -------
    torch.Tensor
        A positive definite covariance matrix of the same shape as the input.

    Raises
    ------
    ValueError
        If cov is not a 2D or 3D tensor.
    RuntimeError
        If the matrix cannot be made positive definite after all attempts.
    """
    if cov.ndim not in (2, 3):
        msg = f"cov must be a 2D or 3D tensor, got shape {tuple(cov.shape)}"
        raise ValueError(msg)

    # If already positive definite, return as is.
    try:
        torch.linalg.cholesky(cov)
        return cov
    except RuntimeError:
        pass

    eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
    if len(cov.shape) == 3:
        eye = eye.unsqueeze(0)

    # Symmetrize first. This is a cheap, silent repair for small numerical
    # asymmetries and keeps the subsequent diagonal jitter symmetric.
    cov = (cov + cov.transpose(-1, -2)) / 2

    # If that fails, try adding increasing amounts of jitter.
    jitter = min_jitter
    for i in range(max_tries):
        cov = cov + jitter * eye

        try:
            torch.linalg.cholesky(cov)
            if i > 0:
                warnings.warn(
                    f"cov not p.d. - added {jitter:.1e} to the diagonal and "
                    "symmetrized",
                    NumericalWarning,
                    stacklevel=2,
                )
            return cov
        except RuntimeError as e:
            if i == max_tries - 1 and not clamp_eigvals:
                msg = (
                    f"Matrix could not be made positive definite after "
                    f"{max_tries} attempts with jitter up to {jitter:.1e}:\n{cov}"
                )
                raise RuntimeError(msg) from e
            # Increase jitter for next attempt.
            jitter *= 10

    # If adding jitter fails, optionally clamp eigvals to find closest positive definite
    # matrix using heuristic for upper bound of eigenvalues if clamping lower bound is
    # not sufficient.
    MIN_EIGVAL = 1e-3
    if clamp_eigvals:
        # Attempt to first only clamp minimum eigenvals.
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=MIN_EIGVAL)
        cov = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
        cov = (cov + cov.transpose(-1, -2)) / 2
        try:
            torch.linalg.cholesky(cov)
            warnings.warn(
                f"cov not p.d. - clamped min eigval to {MIN_EIGVAL:.1e} and "
                "symmetrized; eigenvalue clamping can invalidate gradients through "
                "the repaired covariance",
                NumericalWarning,
                stacklevel=2,
            )
            return cov
        except RuntimeError:
            # Clamp both minimum and maximum eigvals.
            cov = (cov + cov.transpose(-1, -2)) / 2
            eigvals, eigvecs = torch.linalg.eigh(cov)

            # Ensure condition number around 10**6.
            min_eigval = max(torch.min(eigvals).item(), MIN_EIGVAL)
            max_eigval = min(torch.max(eigvals).item(), 1e6 * min_eigval)

            # Clamp eigvals.
            eigvals = torch.clamp(eigvals, min=min_eigval, max=max_eigval)
            cov = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
            cov = (cov + cov.transpose(-1, -2)) / 2
            try:
                torch.linalg.cholesky(cov)
                warnings.warn(
                    f"cov not p.d. - clamped min eigval to {min_eigval:.1e} and max "
                    f"eigval to {max_eigval:.1e}, then symmetrized; eigenvalue "
                    "clamping can invalidate gradients through the repaired covariance",
                    NumericalWarning,
                    stacklevel=2,
                )
                return cov
            except RuntimeError as e:
                msg = (
                    f"Matrix could not be made positive definite after clamping "
                    f"eigenvalues to min={min_eigval:.1e} and max={max_eigval:.1e}:\n"
                    f"{cov}"
                )
                raise RuntimeError(msg) from e

    raise RuntimeError(f"Matrix could not be made positive definite:\n{cov}")
