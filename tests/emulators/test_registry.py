"""Tests for the Registry functionality."""

import pytest
from autoemulate.emulators import Registry
from autoemulate.emulators.base import Emulator, GaussianProcessEmulator


def test_register_custom_emulator():
    """Test registering a custom emulator subclass."""
    registry = Registry()

    class TestEmulator(Emulator): ...

    # Register
    registry.register_model(TestEmulator, overwrite=True)

    # Check it was registered and can be retrieved
    assert TestEmulator in registry.all_emulators
    retrieved = registry.get_emulator_class("TestEmulator")
    assert retrieved == TestEmulator


def test_register_gp_from_factory():
    """Test registering a GP subclass created manually."""
    registry = Registry()

    # Create a custom GP emulator class
    class TestGP(GaussianProcessEmulator): ...

    # Register
    registry.register_model(TestGP, overwrite=True)

    # Check it was registered in the correct lists
    assert TestGP in registry.all_emulators
    assert TestGP in registry.gaussian_process_emulators


def test_overwrite_flag():
    """Test that overwrite flag controls duplicate registration."""
    registry = Registry()

    class TestOverwrite(Emulator):
        version = 1

        @classmethod
        def model_name(cls):
            return "TestOverwrite"

    # Register first version
    registry.register_model(TestOverwrite, overwrite=True)

    # Create second version with same name
    class TestOverwrite2(Emulator):
        version = 2

        @classmethod
        def model_name(cls):
            return "TestOverwrite"

    # Overwrite should succeed
    registry.register_model(TestOverwrite2, overwrite=True)
    retrieved = registry.get_emulator_class("TestOverwrite")
    assert retrieved.version == 2  # type: ignore as version exists if test passes

    # Overwrite=False should raise error
    class TestOverwrite3(Emulator):
        @classmethod
        def model_name(cls):
            return "TestOverwrite"

    with pytest.raises(ValueError, match="already exists"):
        registry.register_model(TestOverwrite3, overwrite=False)
