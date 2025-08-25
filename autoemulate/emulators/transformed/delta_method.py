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
    """
    Delta method for uncertainty propagation through nonlinear functions.

    Supports multidimensional input and output tensors by flattening and unflattening.

    Computes mean and variance of f(X) where X ~ N(μ, Σ) using Taylor expansion:
    - E[f(X)] ≈ f(μ) + (1/2) * tr(H_f(μ) * Σ)  [second-order mean]
    - Var[f(X)] ≈ ∇f(μ)^T * Σ * ∇f(μ)         [first-order variance]

    Parameters
    ----------
    forward_fn: Callable[[TensorLike], TensorLike]
        Function to transform. Can accept multidimensional tensors.
        Expected input shape: (batch_size, *input_shape)
        Expected output shape: (batch_size, *output_shape)
    x_mean: TensorLike
        Input means, shape (batch_size, *input_shape)
    x_variance: TensorLike
        Input variances (diagonal covariance), shape (batch_size, *input_shape)
    include_second_order: bool
        Whether to include second-order mean correction

    Returns
    -------
    dict[str, TensorLike]
        - 'mean_first_order': f(μ) with original output shape
        - 'mean_second_order': Second-order correction with original output shape
        - 'mean_total': Total mean approximation with original output shape
        - 'variance_approx': Variance approximation with original output shape
    """
    if x_mean.shape != x_variance.shape:
        msg = "x_mean and x_variance must have same shape"
        raise ValueError(msg)

    if x_mean.dim() < 1:
        msg = f"Expected at least 1D tensors, got {x_mean.dim()}D"
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
    x_variance: TensorLike,
    original_input_shape: torch.Size,
    include_second_order: bool,
) -> dict[str, TensorLike]:
    """Core delta method computation with flattening support."""
    batch_size = original_input_shape[0]

    # Flatten inputs to 2D: (batch_size, input_dim)
    x_mean_flat = x_mean.reshape(batch_size, -1)
    x_variance_flat = x_variance.reshape(batch_size, -1)
    input_dim = x_mean_flat.shape[1]

    # Create wrapper function that handles reshape for forward_fn
    def forward_fn_flat(x_flat: TensorLike) -> TensorLike:
        """Reshape flat input to original shape, call forward_fn, flatten output."""
        # Reshape from flat to original input shape
        batch_size_inner = x_flat.shape[0]
        original_shape_batch = (batch_size_inner,) + original_input_shape[1:]
        x_reshaped = x_flat.reshape(original_shape_batch)

        # Call original function and flatten output
        return forward_fn(x_reshaped).reshape(batch_size_inner, -1)

    # First-order mean: f(μ) using flattened wrapper
    mean_first_order_flat = forward_fn_flat(x_mean_flat)

    if mean_first_order_flat.dim() == 1:
        mean_first_order_flat = mean_first_order_flat.unsqueeze(-1)

    output_dim = mean_first_order_flat.shape[-1]

    # Get original output shape by calling forward_fn once
    sample_output = forward_fn(x_mean[:1])  # Single sample to get shape
    original_output_shape = sample_output.shape

    # Compute jacobians and hessians using vmap for efficiency
    def compute_delta_terms_single(
        x_single: TensorLike, x_var_single: TensorLike
    ) -> tuple[TensorLike, TensorLike]:
        """Compute variance and second-order mean for single sample."""
        # Jacobian computation
        jacobian_fn = jacrev(forward_fn_flat)
        jac = jacobian_fn(x_single.unsqueeze(0))
        if isinstance(jac, tuple):
            jac = jac[0]
        jac = jac.squeeze(0)
        if jac.dim() == 1:
            jac = jac.unsqueeze(0)

        # Variance: sum over input dimensions of (∂f/∂xi)² * Var(Xi)
        variance = torch.sum(jac.pow(2) * x_var_single.unsqueeze(0), dim=-1)

        # Second-order mean correction
        if include_second_order:

            def single_output_fn(x: TensorLike) -> TensorLike:
                return forward_fn_flat(x.unsqueeze(0)).squeeze(0)

            hessian_fn = hessian(single_output_fn)
            hess = hessian_fn(x_single)

            if isinstance(hess, tuple):
                hess = hess[0]

            if hess.dim() == 2:  # Single output case
                hess = hess.unsqueeze(0)

            # Extract diagonal elements for each output
            diag_elements = []
            for i in range(min(output_dim, hess.shape[0])):
                diag_elements.append(torch.diag(hess[i]))

            if diag_elements:
                hess_diag = torch.stack(diag_elements)
            else:
                hess_diag = torch.zeros(
                    output_dim, input_dim, dtype=x_single.dtype, device=x_single.device
                )

            # Second-order: (1/2) * Σi Hii * Var(Xi)
            second_order = 0.5 * torch.sum(
                hess_diag * x_var_single.unsqueeze(0), dim=-1
            )
        else:
            second_order = torch.zeros(
                output_dim, dtype=x_single.dtype, device=x_single.device
            )

        return variance, second_order

    # Vectorize over batch dimension
    batched_delta_fn = vmap(compute_delta_terms_single)
    variances_flat, second_order_corrections_flat = batched_delta_fn(
        x_mean_flat, x_variance_flat
    )

    # Reshape to match mean shape
    if variances_flat.dim() > 2:
        variances_flat = variances_flat.squeeze(-1)
    if second_order_corrections_flat.dim() > 2:
        second_order_corrections_flat = second_order_corrections_flat.squeeze(-1)

    # Combine results
    mean_total_flat = mean_first_order_flat + second_order_corrections_flat

    # Calculate the correct output shape for the full batch
    # original_output_shape is for 1 sample, we need to expand for full batch
    full_batch_output_shape = (batch_size,) + original_output_shape[1:]

    # Reshape results back to original output shape
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
