import torch
from autoemulate.core.types import TensorLike
from autoemulate.transforms.discrete_fourier import DiscreteFourierTransform


def create_test_data():
    """Create consistent test data for all tests."""
    torch.manual_seed(42)
    n_samples, n_features = 10, 16  # Use power of 2 for cleaner FFT
    n_components = 3
    x = torch.randn(n_samples, n_features)
    return x, n_samples, n_features, n_components


def test_transform_shapes():
    """Test that forward and inverse transforms produce correct shapes."""
    x, n_samples, n_features, n_components = create_test_data()

    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    # Test forward transform shape - should have 2 * n_selected frequencies
    y = dft(x)
    n_selected = len(dft.selected_indices)
    expected_shape = (n_samples, 2 * n_selected)
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"

    # Test inverse transform shape
    x_reconstructed = dft.inv(y)
    assert isinstance(x_reconstructed, TensorLike)
    assert x_reconstructed.shape == x.shape, (
        f"Expected shape {x.shape}, got {x_reconstructed.shape}"
    )


def test_fft_consistency():
    """Test that our implementation is consistent with PyTorch FFT."""
    x, n_samples, n_features, n_components = create_test_data()

    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    # Apply our transform
    y_transform = dft(x)

    # Apply PyTorch's FFT and extract the same components
    x_fft = torch.fft.fft(x, dim=-1)
    selected_fft = x_fft[:, dft.selected_indices]

    # Convert to real/imag pairs like our transform
    real_parts = selected_fft.real
    imag_parts = selected_fft.imag
    y_manual = torch.stack([real_parts, imag_parts], dim=-1).reshape(n_samples, -1)

    # Should be very close (accounting for floating point precision)
    assert torch.allclose(y_transform, y_manual, atol=1e-5), (
        "Transform doesn't match manual FFT implementation"
    )

    # Test inverse consistency
    x_reconstructed = dft.inv(y_transform)
    assert isinstance(x_reconstructed, TensorLike)

    # Manual inverse: zero-pad and IFFT
    full_fft = torch.zeros(n_samples, n_features, dtype=torch.complex64)
    full_fft[:, dft.selected_indices] = selected_fft
    x_manual = torch.fft.ifft(full_fft, dim=-1).real

    assert torch.allclose(x_reconstructed, x_manual, atol=1e-5), (
        "Inverse transform doesn't match manual IFFT implementation"
    )


def test_real_valued_output():
    """Test that all outputs are real-valued (no complex numbers)."""
    x, n_samples, n_features, n_components = create_test_data()

    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    y = dft(x)
    x_reconstructed = dft.inv(y)

    assert y.dtype == torch.float32, (
        f"Transform output should be float32, got {y.dtype}"
    )
    if x_reconstructed is not None:
        assert x_reconstructed.dtype == torch.float32, (
            f"Reconstructed output should be float32, got {x_reconstructed.dtype}"
        )
        assert not torch.is_complex(x_reconstructed), (
            "Reconstructed output should not be complex"
        )
    assert not torch.is_complex(y), "Transform output should not be complex"


def test_forward_shape():
    """Test forward_shape method returns correct output shape."""
    x, n_samples, n_features, n_components = create_test_data()

    dft = DiscreteFourierTransform(n_components=n_components)

    # Test after fitting - forward_shape requires fitting
    input_shape = torch.Size([n_samples, n_features])
    dft.fit(x)
    forward_shape = dft.forward_shape(input_shape)

    # Forward shape should preserve batch dimensions and change last dimension
    assert len(forward_shape) == len(input_shape)
    assert forward_shape[:-1] == input_shape[:-1]  # Batch dims unchanged

    # Test actual shape
    y = dft(x)

    # Verify forward_shape matches actual output shape
    actual_forward_shape = dft.forward_shape(input_shape)
    assert actual_forward_shape == y.shape, (
        f"forward_shape {actual_forward_shape} != actual shape {y.shape}"
    )


def test_inverse_shape():
    """Test inverse_shape method returns correct output shape."""
    x, n_samples, n_features, n_components = create_test_data()

    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    y = dft(x)
    x_reconstructed = dft.inv(y)

    # Test inverse_shape
    inverse_shape = dft.inverse_shape(y.shape)

    # Verify inverse_shape matches actual reconstruction shape
    if x_reconstructed is not None:
        assert inverse_shape == x_reconstructed.shape, (
            f"inverse_shape {inverse_shape} != actual shape {x_reconstructed.shape}"
        )

    # Should match original input shape
    assert inverse_shape == x.shape, (
        f"inverse_shape {inverse_shape} != original shape {x.shape}"
    )


def test_shape_methods_consistency():
    """Test that forward_shape and inverse_shape are consistent."""
    x, n_samples, n_features, n_components = create_test_data()

    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    input_shape = x.shape
    forward_shape = dft.forward_shape(input_shape)
    reconstructed_shape = dft.inverse_shape(forward_shape)

    # inverse_shape(forward_shape(input_shape)) should equal input_shape
    assert reconstructed_shape == input_shape, (
        f"Shape consistency failed: {input_shape} -> {forward_shape} -> "
        f"{reconstructed_shape}"
    )


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
    """Test FFT-based DFT matches PyTorch's FFT implementation."""
    x, n_samples, n_features, n_components = create_test_data()

    # Fit the transform
    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    # Get our transform output
    y_our = dft(x)

    # Apply PyTorch's FFT to the same data
    x_fft = torch.fft.fft(x, dim=1)

    # Extract the same frequency components that our transform selected
    selected_fft = x_fft[:, dft.selected_indices]

    # Convert to the same format as our implementation
    real_parts = selected_fft.real
    imag_parts = selected_fft.imag
    stacked = torch.stack([real_parts, imag_parts], dim=-1)
    y_expected = stacked.reshape(n_samples, -1)

    # Compare the results
    max_error = torch.max(torch.abs(y_our - y_expected))

    # Should be essentially identical (within floating point precision)
    assert max_error < 1e-6, f"Our DFT differs from expected FFT result: {max_error}"
