import torch
from autoemulate.experimental.transforms.discrete_fourier import (
    DiscreteFourierTransform,
)
from autoemulate.experimental.types import TensorLike


def create_test_data():
    """Create consistent test data for all tests."""
    torch.manual_seed(42)
    n_samples, n_features = 10, 8
    n_components = 3
    x = torch.randn(n_samples, n_features)
    return x, n_samples, n_features, n_components


def test_transform_shapes():
    """Test that forward and inverse transforms produce correct shapes."""
    x, n_samples, n_features, n_components = create_test_data()

    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    # Test forward transform shape
    y = dft(x)
    expected_shape = (n_samples, 2 * n_components)
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"

    # Test inverse transform shape
    x_reconstructed = dft.inv(y)
    assert isinstance(x_reconstructed, TensorLike)
    assert x_reconstructed.shape == x.shape, (
        f"Expected shape {x.shape}, got {x_reconstructed.shape}"
    )


def test_basis_matrix_properties():
    """Test that the basis matrix has correct properties."""
    x, n_samples, n_features, n_components = create_test_data()

    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    A = dft._basis_matrix.T
    expected_shape = (2 * n_components, n_features)

    assert A.shape == expected_shape, (
        f"Expected basis matrix shape {expected_shape}, got {A.shape}"
    )
    assert A.dtype == torch.float32, f"Expected float32 dtype, got {A.dtype}"


def test_matrix_multiplication_consistency():
    """Test that transforms work correctly via matrix multiplication."""
    x, n_samples, n_features, n_components = create_test_data()

    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    A = dft._basis_matrix.T

    # Test forward transform via matrix multiplication
    y_transform = dft(x)
    y_manual = x @ A.T

    assert torch.allclose(y_transform, y_manual, atol=1e-6), (
        "Forward transform doesn't match manual matrix multiplication"
    )

    # Test inverse transform via matrix multiplication
    x_reconstructed = dft.inv(y_transform)
    x_manual = y_transform @ A
    assert isinstance(x_reconstructed, TensorLike)
    assert torch.allclose(x_reconstructed, x_manual, atol=1e-6), (
        "Inverse transform doesn't match manual matrix multiplication"
    )


def test_real_valued_output():
    """Test that all outputs are real-valued (no complex numbers)."""
    x, n_samples, n_features, n_components = create_test_data()

    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    y = dft(x)
    A = dft._basis_matrix

    assert y.dtype == torch.float32, (
        f"Transform output should be float32, got {y.dtype}"
    )
    assert A.dtype == torch.float32, f"Basis matrix should be float32, got {A.dtype}"
    assert not torch.is_complex(y), "Transform output should not be complex"
    assert not torch.is_complex(A), "Basis matrix should not be complex"


def test_frequency_component_pairing():
    """Test that frequency components are properly paired as real/imaginary columns."""
    x, n_samples, n_features, n_components = create_test_data()

    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    y = dft(x)

    # Output should have 2*n_components columns (real/imag pairs)
    expected_cols = 2 * n_components
    assert y.shape[1] == expected_cols, (
        f"Expected {expected_cols} columns, got {y.shape[1]}"
    )

    # Should have even number of columns (pairs)
    assert y.shape[1] % 2 == 0, (
        "Output should have even number of columns for real/imag pairs"
    )


def test_against_torch_fft():
    """Test matrix-based DFT against PyTorch's FFT implementation."""
    x, n_samples, n_features, n_components = create_test_data()

    # Fit the transform to get selected frequency components
    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    # Get the selected frequency indices
    freq_indices = dft.freq_indices

    # Apply our matrix-based transform
    y_matrix = dft(x)

    # Apply PyTorch's FFT to the same data
    x_fft = torch.fft.fft(x, dim=1)  # FFT along feature dimension

    # Extract the same frequency components that our transform selected
    selected_fft = x_fft[:, freq_indices]  # Shape: (n_samples, n_components)

    # Convert complex FFT output to real/imag pairs format
    # PyTorch FFT gives complex numbers, we need [real, imag, real, imag, ...]
    fft_real = selected_fft.real  # Shape: (n_samples, n_components)
    fft_imag = selected_fft.imag  # Shape: (n_samples, n_components)

    # Interleave real and imaginary parts to match our format
    y_fft_paired = torch.stack([fft_real, fft_imag], dim=2).reshape(
        n_samples, 2 * n_components
    )

    # Account for normalization difference
    # Our DFT uses 1/sqrt(N) normalization, PyTorch's doesn't normalize by default
    normalization_factor = 1.0 / torch.sqrt(
        torch.tensor(n_features, dtype=torch.float32)
    )
    y_fft_normalized = y_fft_paired * normalization_factor

    # Compare the results
    max_error = torch.max(torch.abs(y_matrix - y_fft_normalized))
    relative_error = max_error / torch.max(torch.abs(y_fft_normalized))

    # Should be very close (accounting for floating point precision)
    assert max_error < 1e-5, (
        f"Matrix DFT differs too much from PyTorch FFT: {max_error}"
    )
    assert relative_error < 1e-4, f"Relative error too large: {relative_error}"
