from collections.abc import Callable

import torch
from torch.func import hessian, jacrev, vmap

from autoemulate.core.types import TensorLike


def delta_method(
    forward_fn: Callable[[TensorLike], TensorLike],
    x_mean: TensorLike,
    x_variance: TensorLike,
    include_second_order: bool = True,
) -> dict[str, TensorLike]:
    """Delta method for uncertainty propagation through nonlinear functions.

    Supports multidimensional input and output tensors by flattening and
    unflattening, and accepts either elementwise variances (diagonal
    covariance) or full covariance matrices per batch element.

    Computes mean and variance of f(X) where X ~ N(μ, Σ) using a Taylor
    expansion around μ:
    - E[f(X)] ≈ f(μ) + (1/2) · tr(H_f(μ) · Σ)        [second-order mean]
    - Var[f(X)] ≈ ∇f(μ)^T · Σ · ∇f(μ)                [first-order variance]

    Parameters
    ----------
    forward_fn: Callable[[TensorLike], TensorLike]
        Function to transform. Can accept multidimensional tensors.
        Expected input shape: `(batch_size, *input_shape)`.
        Expected output shape: `(batch_size, *output_shape)`.
    x_mean: TensorLike
        Input means, shape `(batch_size, *input_shape)`.
    x_variance: TensorLike
        Either elementwise variances (interpreted as diagonal covariance) with
        the same shape as `x_mean`; or full covariance matrices with shape
        `(batch_size, input_dim, input_dim)`, where `input_dim` is the
        product of `*input_shape`.
    include_second_order: bool
        Whether to include the second-order mean correction.

    Returns
    -------
    dict[str, TensorLike]
        - `mean_first_order`: f(μ) with original output shape.
        - `mean_second_order`: Second-order correction with original output
          shape (zeros if `include_second_order=False`).
        - `mean_total`: Total mean approximation with original output shape.
        - `variance_approx`: Variance approximation with original output
          shape, propagated using either diagonal variances or full covariances
          as provided.
    """
    if x_mean.dim() < 1:
        msg = f"Expected at least 1D tensors, got {x_mean.dim()}D"
        raise ValueError(msg)

    # Ensure shape of mean and variance are the same when equal dimensionality
    if x_variance.dim() == x_mean.dim() and x_variance.shape != x_mean.shape:
        msg = "x_mean and x_variance must have same shape"
        raise ValueError(msg)

    # Store original shapes
    original_input_shape = x_mean.shape

    # Handle 1D case by adding batch dimension
    if x_mean.dim() == 1:
        x_mean = x_mean.unsqueeze(0)
        x_variance = x_variance.unsqueeze(0)
        original_input_shape = x_mean.shape

    # Use helper function for the core computation
    return _delta_method_core(
        forward_fn, x_mean, x_variance, original_input_shape, include_second_order
    )


def _delta_method_core(  # noqa: PLR0915
    forward_fn: Callable,
    x_mean: TensorLike,
    x_uncertainty: TensorLike,  # can be diag variances or full covariance
    original_input_shape: torch.Size,
    include_second_order: bool,
) -> dict[str, TensorLike]:
    """Core delta method with support for diagonal or full covariance.

    Parameters
    ----------
    forward_fn:
        See `delta_method`.
    x_mean:
        See `delta_method`.
    x_uncertainty:
        Either elementwise variances in any shape matching the number of
        elements in `x_mean` (interpreted as diagonal covariance), or full
        covariances with shape `(batch_size, input_dim, input_dim)`.
    original_input_shape:
        Original input shape before flattening, including batch dimension.
    include_second_order:
        Whether to include the second-order mean correction.

    Returns
    -------
    dict[str, TensorLike]
        Same keys and shapes as returned by `delta_method`.
    """
    batch_size = original_input_shape[0]

    # Flatten inputs to 2D: (batch_size, input_dim)
    x_mean_flat = x_mean.reshape(batch_size, -1)
    input_dim = x_mean_flat.shape[1]

    # Detect whether x_uncertainty is variances (diag) or covariances (full)
    total_elems = x_uncertainty.numel()
    diag_elems = batch_size * input_dim
    full_elems = batch_size * input_dim * input_dim

    if total_elems == diag_elems:
        cov_type = "diag"
        x_uncertainty_flat = x_uncertainty.reshape(batch_size, input_dim)
    elif total_elems == full_elems and x_uncertainty.dim() >= 3:
        cov_type = "full"
        x_uncertainty_flat = x_uncertainty.reshape(batch_size, input_dim, input_dim)
    else:
        msg = (
            "x_uncertainty must represent either diagonal variances (same number of"
            " elements as x_mean) or full covariances with shape (batch, input_dim,"
            " input_dim)."
        )
        raise ValueError(msg)

    # Create wrapper function that handles reshape for forward_fn
    def forward_fn_flat(x_flat: TensorLike) -> TensorLike:
        batch_size_inner = x_flat.shape[0]
        original_shape_batch = (batch_size_inner,) + original_input_shape[1:]
        x_reshaped = x_flat.reshape(original_shape_batch)
        return forward_fn(x_reshaped).reshape(batch_size_inner, -1)

    # First-order mean: f(μ)
    mean_first_order_flat = forward_fn_flat(x_mean_flat)
    output_dim = mean_first_order_flat.shape[-1]
    original_output_shape = forward_fn(x_mean[:1]).shape

    def compute_delta_terms_single(x_single: TensorLike, unc_single: TensorLike):
        # Define a single-sample function to obtain (output_dim, input_dim) Jacobian
        def single_output_fn(x: TensorLike) -> TensorLike:
            # Using vmap computes without batch but forward_fn_flat expects batch
            return forward_fn_flat(x.unsqueeze(0)).squeeze(0)

        jacobian_fn = jacrev(single_output_fn)

        # (output_dim, input_dim) for vector output; (input_dim,) for scalar
        jac = jacobian_fn(x_single)
        jac = jac[0] if isinstance(jac, tuple) else jac  # handle tuple return
        assert jac.numel() == output_dim * input_dim, "Size mismatch"
        # Ensure correct shape for scalar/vector output
        jac = jac.reshape(output_dim, input_dim)

        # Variance / covariance propagation
        if cov_type == "diag":
            variance = torch.sum(jac.pow(2) * unc_single.unsqueeze(0), dim=-1)
        else:  # full covariance
            cov_out = jac @ unc_single @ jac.T  # (output_dim, output_dim)
            variance = torch.diagonal(cov_out, dim1=-2, dim2=-1)

        # Second-order mean correction
        if include_second_order:
            hessian_fn = hessian(single_output_fn)
            hess = hessian_fn(x_single)  # (output_dim, input_dim, input_dim)
            hess = hess[0] if isinstance(hess, tuple) else hess  # handle tuple return
            # Ensure correct shape for scalar/vector output
            assert hess.numel() == output_dim * input_dim * input_dim, "Size mismatch"
            hess = hess.reshape(output_dim, input_dim, input_dim)

            if cov_type == "diag":
                second_order = 0.5 * torch.sum(
                    torch.diagonal(hess, dim1=-2, dim2=-1) * unc_single.unsqueeze(0),
                    dim=-1,
                )
            else:
                # full covariance: 0.5 * E((x - μ)^T @ H @ (x - μ)) = 0.5 * trace(H @ Σ)
                second_order = 0.5 * torch.einsum("oij,ji->o", hess, unc_single)
        else:
            second_order = torch.zeros(
                output_dim, dtype=x_single.dtype, device=x_single.device
            )

        return variance, second_order

    batched_delta_fn = vmap(compute_delta_terms_single)
    variances_flat, second_order_corrections_flat = batched_delta_fn(
        x_mean_flat, x_uncertainty_flat
    )

    mean_total_flat = mean_first_order_flat + second_order_corrections_flat
    full_batch_output_shape = (batch_size,) + original_output_shape[1:]

    mean_first_order = mean_first_order_flat.reshape(full_batch_output_shape)
    mean_second_order = second_order_corrections_flat.reshape(full_batch_output_shape)
    mean_total = mean_total_flat.reshape(full_batch_output_shape)
    variances = variances_flat.reshape(full_batch_output_shape)

    return {
        "mean_first_order": mean_first_order,
        "mean_second_order": mean_second_order,
        "mean_total": mean_total,
        "variance_approx": variances,
    }
