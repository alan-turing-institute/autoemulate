"""Tests for transform serialization and deserialization."""

import json

import pytest
from autoemulate.experimental.transforms import (
    TRANSFORM_REGISTRY,
    PCATransform,
    StandardizeTransform,
    VAETransform,
)
from autoemulate.experimental.transforms.base import AutoEmulateTransform


def test_pca_serialization():
    """Test PCA transform serialization and deserialization."""
    # Create PCA transform with specific parameters
    original = PCATransform(n_components=5, niter=500, cache_size=1)

    # Serialize
    serialized = original.to_dict()

    # Check format
    assert len(serialized) == 1
    assert "pca" in serialized
    assert isinstance(serialized["pca"], dict)

    # Check parameters
    params = serialized["pca"]
    assert params["n_components"] == 5
    assert params["niter"] == 500
    assert params["cache_size"] == 1

    # Test deserialization
    restored = AutoEmulateTransform.from_dict(serialized)
    assert isinstance(restored, PCATransform)
    assert restored.n_components == original.n_components
    assert restored.niter == original.niter


def test_standardize_serialization():
    """Test Standardize transform serialization and deserialization."""
    # Create Standardize transform
    original = StandardizeTransform()

    # Serialize
    serialized = original.to_dict()

    # Check format
    assert len(serialized) == 1
    assert "standardize" in serialized
    assert isinstance(serialized["standardize"], dict)

    # Standardize has no parameters, so should be empty dict
    assert serialized["standardize"] == {}

    # Test deserialization
    restored = AutoEmulateTransform.from_dict(serialized)
    assert isinstance(restored, StandardizeTransform)


def test_vae_serialization():
    """Test VAE transform serialization and deserialization."""
    # Create VAE transform with specific parameters
    original = VAETransform(
        latent_dim=3,
        hidden_layers=[64, 32],
        epochs=10,
        batch_size=16,
        learning_rate=1e-3,
        random_seed=42,
        beta=0.8,
        verbose=False,
        cache_size=0,
    )

    # Serialize
    serialized = original.to_dict()

    # Check format
    assert len(serialized) == 1
    assert "vae" in serialized
    assert isinstance(serialized["vae"], dict)

    # Check parameters
    params = serialized["vae"]
    assert params["latent_dim"] == 3
    assert params["hidden_layers"] == [64, 32]
    assert params["epochs"] == 10
    assert params["batch_size"] == 16
    assert params["learning_rate"] == 1e-3
    assert params["random_seed"] == 42
    assert params["beta"] == 0.8
    assert params["verbose"] is False
    assert params["cache_size"] == 0

    # Test deserialization
    restored = AutoEmulateTransform.from_dict(serialized)
    assert isinstance(restored, VAETransform)
    assert restored.latent_dim == original.latent_dim
    assert restored.hidden_layers == original.hidden_layers
    assert restored.epochs == original.epochs
    assert restored.batch_size == original.batch_size
    assert restored.learning_rate == original.learning_rate
    assert restored.random_seed == original.random_seed
    assert restored.beta == original.beta
    assert restored.verbose == original.verbose


def test_json_compatibility():
    """Test that serialized transforms can be converted to/from JSON."""
    transforms = [
        PCATransform(n_components=3),
        StandardizeTransform(),
        VAETransform(latent_dim=2, hidden_layers=[16], epochs=5),
    ]

    for transform in transforms:
        # Serialize to dict
        serialized = transform.to_dict()

        # Convert to JSON string and back
        json_str = json.dumps(serialized)
        restored_dict = json.loads(json_str)

        # Deserialize
        restored_transform = AutoEmulateTransform.from_dict(restored_dict)

        # Check type matches
        assert isinstance(restored_transform, type(transform))


def test_registry_contains_all_transforms():
    """Test that the registry contains all expected transforms."""
    expected_transforms = {"pca", "standardize", "vae"}
    assert set(TRANSFORM_REGISTRY) == expected_transforms

    # Check that all registry values are valid transform classes
    for name, transform_class in TRANSFORM_REGISTRY.items():
        assert issubclass(transform_class, AutoEmulateTransform)
        # Check that the name mapping is correct
        class_name = transform_class.__name__
        if class_name.endswith("Transform"):
            expected_name = class_name[:-9].lower()
        else:
            expected_name = class_name.lower()
        assert name == expected_name


def test_unknown_transform_raises_error():
    """Test that unknown transform names raise ValueError."""
    with pytest.raises(ValueError, match="Unknown transform"):
        AutoEmulateTransform.from_dict({"unknown_transform": {}})


def test_invalid_dict_format_multiple_transforms():
    """Test error when dict contains multiple transforms."""
    invalid_dict = {"pca": {"n_components": 3}, "vae": {"latent_dim": 2}}

    with pytest.raises(
        ValueError, match="Dictionary must contain exactly one transform"
    ):
        AutoEmulateTransform.from_dict(invalid_dict)


def test_invalid_dict_format_empty():
    """Test error when dict is empty."""
    empty_dict = {}

    with pytest.raises(
        ValueError, match="Dictionary must contain exactly one transform"
    ):
        AutoEmulateTransform.from_dict(empty_dict)


def test_invalid_parameters():
    """Test error when deserializing with invalid parameters."""
    # This should fail because n_components is required for PCA
    invalid_dict = {"pca": {"invalid_param": 123}}

    with pytest.raises(TypeError):
        AutoEmulateTransform.from_dict(invalid_dict)
