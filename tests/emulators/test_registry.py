import pytest
from autoemulate.emulators import Registry
from autoemulate.emulators.base import Emulator, GaussianProcessEmulator


@pytest.fixture
def registry():
    return Registry()


def test_register_custom_emulator(registry):
    """Test registering a custom emulator subclass."""

    class TestEmulator(Emulator): ...

    # Register
    registry.register_model(TestEmulator, overwrite=True)

    # Check it was registered and can be retrieved
    assert TestEmulator in registry.all_emulators
    retrieved = registry.get_emulator_class("TestEmulator")
    assert retrieved == TestEmulator


def test_register_gp_from_factory(registry):
    """Test registering a GP subclass created manually."""

    # Create a custom GP emulator class
    class TestGP(GaussianProcessEmulator): ...

    # Register
    registry.register_model(TestGP, overwrite=True)

    # Check it was registered in the correct lists
    assert TestGP in registry.all_emulators
    assert TestGP in registry.gaussian_process_emulators


def test_overwrite_flag(registry):
    """Test that overwrite flag controls duplicate registration."""

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


@pytest.mark.parametrize(
    "emulator_cls", Registry().all_emulators, ids=lambda cls: cls.model_name()
)
def test_get_emulator_class_by_name(registry, emulator_cls):
    """Test that all registered emulators can be retrieved by model_name."""
    model_name = emulator_cls.model_name()

    # Should retrieve the correct class by model_name (case-insensitive)
    assert registry.get_emulator_class(model_name) == emulator_cls
    assert registry.get_emulator_class(model_name.upper()) == emulator_cls
    assert registry.get_emulator_class(model_name.lower()) == emulator_cls


@pytest.mark.parametrize(
    "emulator_cls", Registry().all_emulators, ids=lambda cls: cls.model_name()
)
def test_get_emulator_class_by_short_name(registry, emulator_cls):
    """Test that all registered emulators can be retrieved by short_name."""
    short_name = emulator_cls.short_name()

    # Should retrieve the correct class by short_name (case-insensitive)
    assert registry.get_emulator_class(short_name) == emulator_cls
    assert registry.get_emulator_class(short_name.upper()) == emulator_cls
    assert registry.get_emulator_class(short_name.lower()) == emulator_cls


def test_get_emulator_class_unknown_raises(registry):
    """Test that get_emulator_class raises ValueError for unknown emulator."""

    with pytest.raises(ValueError, match="Unknown emulator name"):
        registry.get_emulator_class("UnknownEmulator")
