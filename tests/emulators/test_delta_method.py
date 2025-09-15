import numpy as np
import pytest
import torch
from autoemulate.emulators.transformed.delta_method import delta_method


def test_linear_function_scalar():
    """Test delta method with scalar linear function (should be exact)."""

    def f(x):
        return 2 * x[:, 0] + 3 * x[:, 1] + 1

    x_mean = torch.tensor([[1.0, 2.0]])
    x_var = torch.tensor([[0.1, 0.2]])

    result = delta_method(f, x_mean, x_var)

    # Linear function: exact results - expected mean = 9.0, variance = 2.2
    assert torch.allclose(result["mean_first_order"], torch.tensor([[9.0]]))
    assert torch.allclose(result["variance_approx"], torch.tensor([[2.2]]))
    assert torch.allclose(result["mean_second_order"], torch.tensor([[0.0]]), atol=1e-6)


def test_quadratic_function_theoretical():
    """Test delta method with quadratic function against theoretical results."""

    def f(x):
        return x[:, 0] ** 2

    x_mean = torch.tensor([[0.0]])
    x_var = torch.tensor([[1.0]])

    result = delta_method(f, x_mean, x_var, include_second_order=True)

    # For f(x) = x^2 around x=0 with variance σ²=1:
    # E[f(X)] ≈ f(0) + (1/2) * f''(0) * σ² = 0 + (1/2) * 2 * 1 = 1
    # Var[f(X)] ≈ (f'(0))² * σ² = 0² * 1 = 0

    assert torch.allclose(result["mean_first_order"], torch.tensor([[0.0]]))
    assert torch.allclose(result["mean_second_order"], torch.tensor([[1.0]]))
    assert torch.allclose(result["mean_total"], torch.tensor([[1.0]]))
    assert torch.allclose(result["variance_approx"], torch.tensor([[0.0]]), atol=1e-6)


def test_exponential_function_first_order():
    """Test delta method with exponential function (first-order only)."""

    def f(x):
        return torch.exp(x[:, 0])

    x_mean = torch.tensor([[0.0]])
    x_var = torch.tensor([[0.1]])

    result = delta_method(f, x_mean, x_var, include_second_order=False)

    # For f(x) = exp(x) around x=0:
    # E[f(X)] ≈ exp(0) = 1
    # Var[f(X)] ≈ (exp(0))² * σ² = 1 * 0.1 = 0.1

    assert torch.allclose(result["mean_first_order"], torch.tensor([[1.0]]))
    assert torch.allclose(result["variance_approx"], torch.tensor([[0.1]]))


def test_constant_function():
    """Test delta method with constant function (variance should be zero)."""

    def f(x):
        return torch.full((x.shape[0], 1), 5.0)

    x_mean = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_var = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    result = delta_method(f, x_mean, x_var)

    assert torch.allclose(result["mean_first_order"], torch.tensor([[5.0], [5.0]]))
    assert torch.allclose(
        result["variance_approx"], torch.tensor([[0.0], [0.0]]), atol=1e-6
    )
    assert torch.allclose(
        result["mean_second_order"], torch.tensor([[0.0], [0.0]]), atol=1e-6
    )


def test_2d_input_sum_function():
    """Test delta method with 2D input (image-like) and sum function."""

    def f(x):
        # x shape: (batch, height, width)
        return torch.sum(x.view(x.shape[0], -1), dim=1, keepdim=True)

    # 2 samples of 3x3 "images"
    x_mean = torch.ones(2, 3, 3)
    x_var = torch.ones(2, 3, 3) * 0.1

    result = delta_method(f, x_mean, x_var)

    # Sum of 9 ones = 9, variance = 9 * 0.1 = 0.9
    expected_mean = torch.tensor([[9.0], [9.0]])
    expected_var = torch.tensor([[0.9], [0.9]])

    assert torch.allclose(result["mean_first_order"], expected_mean)
    assert torch.allclose(result["variance_approx"], expected_var)


def test_3d_input_channel_means():
    """Test delta method with 3D input (RGB image-like) returning channel means."""

    def f(x):
        # x shape: (batch, channels, height, width)
        # Return mean of each channel
        return torch.mean(x.view(x.shape[0], x.shape[1], -1), dim=2)

    # 2 samples of 3x4x4 RGB images
    batch_size, channels, height, width = 2, 3, 4, 4
    x_mean = torch.randn(batch_size, channels, height, width)
    x_var = torch.ones_like(x_mean) * 0.05

    result = delta_method(f, x_mean, x_var)

    # Expected shapes: (batch_size, channels)
    assert result["mean_first_order"].shape == (batch_size, channels)
    assert result["variance_approx"].shape == (batch_size, channels)

    # For mean operation, variance should be reduced by number of pixels
    expected_var_scale = 0.05 / (height * width)
    expected_var = torch.full((batch_size, channels), expected_var_scale)

    assert torch.allclose(result["variance_approx"], expected_var, rtol=1e-5)


def test_4d_input_convolution_like():
    """Test delta method with 4D input simulating a convolution-like operation."""

    def f(x):
        # x shape: (batch, channels, height, width)
        # Simple operation: sum over spatial dimensions, keep channels
        return torch.sum(x, dim=(2, 3))

    batch_size, channels, height, width = 3, 2, 5, 5
    x_mean = torch.ones(batch_size, channels, height, width) * 2.0
    x_var = torch.ones_like(x_mean) * 0.1

    result = delta_method(f, x_mean, x_var)

    # Sum over 5x5=25 pixels, each with mean=2.0, so total mean = 50.0
    expected_mean = torch.full((batch_size, channels), 50.0)
    # Variance: 25 pixels * 0.1 = 2.5
    expected_var = torch.full((batch_size, channels), 2.5)

    assert torch.allclose(result["mean_first_order"], expected_mean)
    assert torch.allclose(result["variance_approx"], expected_var)


def test_1d_input_handling():
    """Test delta method with 1D input (automatic batch dimension addition)."""

    def f(x):
        return x[:, 0] ** 2 + x[:, 1]

    # 1D input (will be unsqueezed to batch size 1)
    x_mean = torch.tensor([2.0, 3.0])
    x_var = torch.tensor([0.1, 0.2])

    result = delta_method(f, x_mean, x_var)

    # Expected: f(2, 3) = 4 + 3 = 7
    # Variance:
    #   (∂f/∂x₀)² * var₀ + (∂f/∂x₁)² * var₁ = (2*2)² * 0.1 + 1² * 0.2 = 1.6 + 0.2 = 1.8
    assert torch.allclose(result["mean_first_order"], torch.tensor([[7.0]]))
    assert torch.allclose(result["variance_approx"], torch.tensor([[1.8]]))


def test_vector_output_function():
    """Test delta method with function returning multiple outputs."""

    def f(x):
        return torch.stack([x[:, 0] ** 2, x[:, 1] * 2, x[:, 0] + x[:, 1]], dim=1)

    x_mean = torch.tensor([[1.0, 2.0]])
    x_var = torch.tensor([[0.1, 0.2]])

    result = delta_method(f, x_mean, x_var)

    # Expected means: [1², 2*2, 1+2] = [1, 4, 3]
    expected_mean = torch.tensor([[1.0, 4.0, 3.0]])

    # Expected variances:
    # For x₀²: (2*x₀)² * var₀ = 4 * 0.1 = 0.4
    # For 2*x₁: 2² * var₁ = 4 * 0.2 = 0.8
    # For x₀+x₁: 1² * var₀ + 1² * var₁ = 0.1 + 0.2 = 0.3
    expected_var = torch.tensor([[0.4, 0.8, 0.3]])

    assert torch.allclose(result["mean_first_order"], expected_mean)
    assert torch.allclose(result["variance_approx"], expected_var)


def test_batch_processing():
    """Test delta method with multiple samples in batch."""

    def f(x):
        return x[:, 0] ** 2 + torch.sin(x[:, 1])

    x_mean = torch.tensor([[1.0, 0.0], [2.0, np.pi / 2], [0.0, np.pi]])
    x_var = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.05, 0.05]])

    result = delta_method(f, x_mean, x_var)

    # Check shapes: the function returns scalar outputs, shape should be (batch_size,)
    assert result["mean_first_order"].shape == (3,)
    assert result["variance_approx"].shape == (3,)
    assert result["mean_second_order"].shape == (3,)

    # Check specific values for first sample
    # f(1, 0) = 1² + sin(0) = 1 + 0 = 1
    assert torch.allclose(result["mean_first_order"][0], torch.tensor(1.0))


def test_large_batch():
    """Test delta method with large batch size."""

    def f(x):
        return torch.sum(x, dim=1, keepdim=True)

    batch_size = 100
    input_dim = 10
    x_mean = torch.randn(batch_size, input_dim)
    x_var = torch.ones_like(x_mean) * 0.1

    result = delta_method(f, x_mean, x_var)

    assert result["mean_first_order"].shape == (batch_size, 1)
    assert result["variance_approx"].shape == (batch_size, 1)

    # For sum function, variance should be sum of input variances
    expected_var = torch.full((batch_size, 1), input_dim * 0.1)
    assert torch.allclose(result["variance_approx"], expected_var)


def test_shape_mismatch_error():
    """Test error when x_mean and x_variance have different shapes."""

    def f(x):
        return x[:, 0]

    x_mean = torch.tensor([[1.0, 2.0]])
    x_var = torch.tensor([[0.1]])  # Wrong shape

    with pytest.raises(ValueError, match="x_uncertainty must be diag variances"):
        delta_method(f, x_mean, x_var)


def test_zero_dimensional_input_error():
    """Test error with zero-dimensional input."""

    def f(x):
        return x

    x_mean = torch.tensor(1.0)  # 0D tensor
    x_var = torch.tensor(0.1)

    with pytest.raises(ValueError, match="at least 1D"):
        delta_method(f, x_mean, x_var)


def test_linearity_property():
    """Test that linear combinations preserve delta method properties."""

    def f1(x):
        return x[:, 0] ** 2

    def f2(x):
        return x[:, 1] ** 2

    def f_combined(x):
        return 2 * f1(x) + 3 * f2(x)

    x_mean = torch.tensor([[1.0, 2.0]])
    x_var = torch.tensor([[0.1, 0.2]])

    result1 = delta_method(f1, x_mean, x_var)
    result2 = delta_method(f2, x_mean, x_var)
    result_combined = delta_method(f_combined, x_mean, x_var)

    # Check linearity of first-order terms
    expected_first_order = (
        2 * result1["mean_first_order"] + 3 * result2["mean_first_order"]
    )
    expected_variance = 4 * result1["variance_approx"] + 9 * result2["variance_approx"]

    assert torch.allclose(result_combined["mean_first_order"], expected_first_order)
    assert torch.allclose(result_combined["variance_approx"], expected_variance)


def test_second_order_inclusion():
    """Test that second-order terms are included when requested."""

    def f(x):
        return x[:, 0] ** 4  # High-order nonlinearity

    x_mean = torch.tensor([[1.0]])  # Test at x=1 where derivatives are non-zero
    x_var = torch.tensor([[0.5]])

    result_first_only = delta_method(f, x_mean, x_var, include_second_order=False)
    result_second_included = delta_method(f, x_mean, x_var, include_second_order=True)

    # First-order only should have zero second-order term
    assert torch.allclose(result_first_only["mean_second_order"], torch.tensor([0.0]))

    # Second-order included should have non-zero second-order term
    # At x=1: f(1)=1, f'(1)=4, f''(1)=12
    # Second-order correction = (1/2) * f''(1) * var = (1/2) * 12 * 0.5 = 3.0
    assert not torch.allclose(
        result_second_included["mean_second_order"], torch.tensor([0.0])
    )

    # Total means should be different
    assert not torch.allclose(
        result_first_only["mean_total"], result_second_included["mean_total"]
    )


def test_small_variance_stability():
    """Test numerical stability with very small variances."""

    def f(x):
        return torch.exp(x[:, 0])

    x_mean = torch.tensor([[1.0]])
    x_var = torch.tensor([[1e-10]])  # Very small variance

    result = delta_method(f, x_mean, x_var)

    # Should not produce NaN or inf values
    assert torch.isfinite(result["mean_first_order"]).all()
    assert torch.isfinite(result["variance_approx"]).all()
    assert torch.isfinite(result["mean_second_order"]).all()


def test_large_variance_stability():
    """Test numerical stability with large variances."""

    def f(x):
        return torch.tanh(x[:, 0])  # Bounded function

    x_mean = torch.tensor([[0.0]])
    x_var = torch.tensor([[100.0]])  # Large variance

    result = delta_method(f, x_mean, x_var)

    # Should not produce NaN or inf values
    assert torch.isfinite(result["mean_first_order"]).all()
    assert torch.isfinite(result["variance_approx"]).all()
    assert torch.isfinite(result["mean_second_order"]).all()


def test_extreme_input_values():
    """Test delta method with extreme input values."""

    def f(x):
        return torch.sigmoid(x[:, 0])  # Sigmoid is numerically stable

    x_mean = torch.tensor([[1000.0], [-1000.0]])  # Extreme values
    x_var = torch.tensor([[1.0], [1.0]])

    result = delta_method(f, x_mean, x_var)

    # Should not produce NaN or inf values
    assert torch.isfinite(result["mean_first_order"]).all()
    assert torch.isfinite(result["variance_approx"]).all()
    assert torch.isfinite(result["mean_second_order"]).all()


def test_single_input_dimension():
    """Test delta method with single input dimension."""

    def f(x):
        return x[:, 0] ** 3 + 2

    x_mean = torch.tensor([[2.0]])
    x_var = torch.tensor([[0.1]])

    result = delta_method(f, x_mean, x_var)

    # f(2) = 8 + 2 = 10
    # f'(2) = 3 * 4 = 12, so variance = 12² * 0.1 = 14.4
    assert torch.allclose(result["mean_first_order"], torch.tensor([10.0]))
    assert torch.allclose(result["variance_approx"], torch.tensor([14.4]))


def test_linear_function_scalar_full_covariance():
    """Scalar linear function with full covariance (off-diagonal tested)."""

    def f(x):
        return 2 * x[:, 0] + 3 * x[:, 1] + 1

    x_mean = torch.tensor([[1.0, 2.0]])
    # Full covariance with correlation term
    # [[0.1, 0.05], [0.05, 0.2]]
    x_cov = torch.tensor([[[0.1, 0.05], [0.05, 0.2]]])

    result = delta_method(f, x_mean, x_cov)

    # Mean is exact for linear
    assert torch.allclose(result["mean_first_order"], torch.tensor([[9.0]]))
    # Var = [2,3] Σ [2,3]^T = 4*0.1 + 2*2*3*0.05 + 9*0.2 = 2.8
    assert torch.allclose(result["variance_approx"], torch.tensor([[2.8]]), atol=1e-6)
    assert torch.allclose(result["mean_second_order"], torch.tensor([[0.0]]), atol=1e-6)


def test_vector_output_function_full_covariance():
    """Vector output with full covariance including correlation."""

    def f(x):
        return torch.stack([x[:, 0] ** 2, x[:, 1] * 2, x[:, 0] + x[:, 1]], dim=1)

    x_mean = torch.tensor([[1.0, 2.0]])
    # Σ with off-diagonal
    x_cov = torch.tensor([[[0.1, 0.05], [0.05, 0.2]]])

    result = delta_method(f, x_mean, x_cov)

    # Means unchanged
    expected_mean = torch.tensor([[1.0, 4.0, 3.0]])
    assert torch.allclose(result["mean_first_order"], expected_mean)

    # Vars are diag(J Σ J^T) at x=[1,2]:
    # y0=x0^2 -> grad [2,0] => 4*0.1 = 0.4
    # y1=2*x1  -> grad [0,2] => 4*0.2 = 0.8
    # y2=x0+x1 -> grad [1,1] => 0.1 + 2*0.05 + 0.2 = 0.4
    expected_var = torch.tensor([[0.4, 0.8, 0.4]])
    assert torch.allclose(result["variance_approx"], expected_var, atol=1e-6)


def test_2d_input_sum_function_full_diag_equivalence():
    """Full diagonal covariance equals elementwise variance case."""

    def f(x):
        return torch.sum(x.view(x.shape[0], -1), dim=1, keepdim=True)

    x_mean = torch.ones(2, 3, 3)
    x_var = torch.ones(2, 3, 3) * 0.1

    # Build full covariance as diagonal from variances
    var_flat = x_var.view(x_var.shape[0], -1)
    x_cov_full = torch.diag_embed(var_flat)
    print(x_mean.shape, x_cov_full.shape, var_flat.shape)
    res_var = delta_method(f, x_mean, x_var)
    res_cov = delta_method(f, x_mean, x_cov_full)

    for key in (
        "mean_first_order",
        "variance_approx",
        "mean_second_order",
        "mean_total",
    ):
        assert torch.allclose(res_var[key], res_cov[key], atol=1e-6)


def test_invalid_full_covariance_shape_error():
    """Error when a 3D tensor is provided but not (batch, D, D)."""

    def f(x):
        return x[:, 0]

    x_mean = torch.tensor([[1.0, 2.0]])
    # Wrong shape: element count does not match diag (B*D) or full (B*D*D)
    # Here with B=1, D=2 => diag=2, full=4; choose 6 elements to force error
    x_cov_wrong = torch.zeros((1, 2, 3))

    with pytest.raises(ValueError, match="either diagonal variances|full covariances"):
        delta_method(f, x_mean, x_cov_wrong)


def test_variance_diff_between_diag_and_full_covariance():
    """Variance should differ when off-diagonal correlations are present."""

    def f(x):
        # Scalar output without keepdim
        return x[:, 0] + x[:, 1]

    x_mean = torch.tensor([[1.0, 2.0]])
    # Diagonal variances
    x_var = torch.tensor([[0.1, 0.2]])
    # Full covariance with positive correlation
    x_cov = torch.tensor([[[0.1, 0.05], [0.05, 0.2]]])

    res_var = delta_method(f, x_mean, x_var)
    res_cov = delta_method(f, x_mean, x_cov)

    # Means equal for linear function
    assert torch.allclose(res_var["mean_first_order"], res_cov["mean_first_order"])

    # Variance differs due to off-diagonal:
    #   diag -> 0.1 + 0.2 = 0.3
    #   full -> 0.1 + 0.2 + 2*0.05 = 0.4
    assert torch.allclose(res_var["variance_approx"], torch.tensor([0.3]), atol=1e-6)
    assert torch.allclose(res_cov["variance_approx"], torch.tensor([0.4]), atol=1e-6)
    assert not torch.allclose(res_var["variance_approx"], res_cov["variance_approx"])
