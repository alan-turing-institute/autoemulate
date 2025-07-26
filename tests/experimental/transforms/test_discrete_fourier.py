import torch
from autoemulate.experimental.transforms.discrete_fourier import (
    DiscreteFourierTransform,
)


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
    assert x_reconstructed.shape == x.shape, (
        f"Expected shape {x.shape}, got {x_reconstructed.shape}"
    )

    print(f"✓ Forward transform: {x.shape} → {y.shape}")
    print(f"✓ Inverse transform: {y.shape} → {x_reconstructed.shape}")


def test_basis_matrix_properties():
    """Test that the basis matrix has correct properties."""
    x, n_samples, n_features, n_components = create_test_data()

    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    A = dft._basis_matrix
    expected_shape = (2 * n_components, n_features)

    assert A.shape == expected_shape, (
        f"Expected basis matrix shape {expected_shape}, got {A.shape}"
    )
    assert A.dtype == torch.float32, f"Expected float32 dtype, got {A.dtype}"

    print(f"✓ Basis matrix shape: {A.shape}")
    print(f"✓ Basis matrix dtype: {A.dtype}")


def test_matrix_multiplication_consistency():
    """Test that transforms work correctly via matrix multiplication."""
    x, n_samples, n_features, n_components = create_test_data()

    dft = DiscreteFourierTransform(n_components=n_components)
    dft.fit(x)

    A = dft._basis_matrix

    # Test forward transform via matrix multiplication
    y_transform = dft(x)
    y_manual = x @ A.T

    assert torch.allclose(y_transform, y_manual, atol=1e-6), (
        "Forward transform doesn't match manual matrix multiplication"
    )

    # Test inverse transform via matrix multiplication
    x_reconstructed = dft.inv(y_transform)
    x_manual = y_transform @ A

    assert torch.allclose(x_reconstructed, x_manual, atol=1e-6), (
        "Inverse transform doesn't match manual matrix multiplication"
    )

    print("✓ Forward transform matches manual computation")
    print("✓ Inverse transform matches manual computation")


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

    print("✓ All outputs are real-valued")


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

    print(
        f"✓ Output has {n_components} frequency components "
        f"as {2 * n_components} real/imag paired columns"
    )


def run_all_tests():
    """Run all test functions."""
    print("Running discrete Fourier transform tests...\n")

    test_transform_shapes()
    print()

    test_basis_matrix_properties()
    print()

    test_matrix_multiplication_consistency()
    print()

    test_real_valued_output()
    print()

    test_frequency_component_pairing()
    print()

    print("All tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()
