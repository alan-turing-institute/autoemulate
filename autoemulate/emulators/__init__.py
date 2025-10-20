from collections.abc import Callable
from typing import overload

from torch import nn

from .base import Emulator, GaussianProcessEmulator
from .ensemble import EnsembleMLP, EnsembleMLPDropout
from .gaussian_process.exact import (
    GaussianProcessCorrelatedMatern32,
    GaussianProcessCorrelatedRBF,
    GaussianProcessMatern32,
    GaussianProcessRBF,
)
from .lightgbm import LightGBM
from .nn.mlp import MLP
from .polynomials import PolynomialRegression
from .radial_basis_functions import RadialBasisFunctions
from .random_forest import RandomForest
from .svm import SupportVectorMachine
from .transformed.base import TransformedEmulator


class Registry:
    """
    Registry for managing emulators.

    The Registry class maintains collections of emulator classes organized by
    their properties (e.g., Gaussian Process emulators, PyTorch-based emulators).
    It provides methods to register new emulators and retrieve them by name.

    Attributes
    ----------
    emulators : dict[str, type[Emulator]]
        Dictionary mapping emulator names to classes.
    """

    emulators: dict[str, type[Emulator]]

    def __init__(self):
        # Initialize the registry with default emulators, this is not updated
        self._default_emulators: list[type[Emulator]] = [
            GaussianProcessMatern32,
            GaussianProcessRBF,
            RadialBasisFunctions,
            PolynomialRegression,
            MLP,
            EnsembleMLP,
        ]

        self._non_pytorch_emulators: list[type[Emulator]] = [
            LightGBM,
            SupportVectorMachine,
            RandomForest,
        ]

        self._all_emulators: list[type[Emulator]] = [
            *self._default_emulators,
            *self._non_pytorch_emulators,
            GaussianProcessCorrelatedMatern32,
            GaussianProcessCorrelatedRBF,
            EnsembleMLPDropout,
        ]

        self._pytorch_emulators: list[type[Emulator]] = [
            emulator
            for emulator in self._all_emulators
            if emulator not in self._non_pytorch_emulators
        ]
        self._gaussian_process_emulators: list[type[Emulator]] = [
            emulator
            for emulator in self._all_emulators
            if issubclass(emulator, GaussianProcessEmulator)
        ]

        self._emulator_registry = {
            em_cls.model_name().lower(): em_cls for em_cls in self._all_emulators
        }
        self._emulator_registry_short_name = {
            em_cls.short_name(): em_cls for em_cls in self._all_emulators
        }

    def register_model(
        self, model_cls: type[Emulator], overwrite: bool = True
    ) -> type[Emulator]:
        """
        Register a new emulator model to the registry.

        Can be used as a method or as a decorator.

        Parameters
        ----------
        model_cls: type[Emulator]
            The emulator class to register.
        overwrite: bool
            If True, allows overwriting an existing model with the same name. If False,
            raises an error if a model with the same name already exists. Defaults to
            True.

        Returns
        -------
        type[Emulator]
            The registered emulator class (unchanged).

        Raises
        ------
        ValueError
            If overwrite is False and a model with the same name already exists.

        """
        model_name = model_cls.model_name().lower()
        short_name = model_cls.short_name()

        # Check if model already exists
        existing_cls_by_name = self._emulator_registry.get(model_name)
        existing_cls_by_short_name = self._emulator_registry_short_name.get(short_name)

        if not overwrite:
            if existing_cls_by_name is not None:
                raise ValueError(
                    f"Model with name '{model_name}' already exists. Set overwrite=True"
                    f" to replace it."
                )
            if existing_cls_by_short_name is not None:
                raise ValueError(
                    f"Model with short name '{short_name}' already exists. Set "
                    f"overwrite=True to replace it."
                )

        # If overwriting, remove the old model from all lists
        if (
            existing_cls_by_name is not None
            and existing_cls_by_name in self._all_emulators
        ):
            self._all_emulators.remove(existing_cls_by_name)
            if existing_cls_by_name in self._gaussian_process_emulators:
                self._gaussian_process_emulators.remove(existing_cls_by_name)
            if existing_cls_by_name in self._pytorch_emulators:
                self._pytorch_emulators.remove(existing_cls_by_name)
            if existing_cls_by_name in self._non_pytorch_emulators:
                self._non_pytorch_emulators.remove(existing_cls_by_name)

        # Add to all_emulators if not already present
        if model_cls not in self._all_emulators:
            self._all_emulators.append(model_cls)

        # Update registries
        self._emulator_registry[model_name] = model_cls
        self._emulator_registry_short_name[short_name] = model_cls

        # Add the gaussian process emulator list if a GaussianProcessEmulator subclass
        if (
            issubclass(model_cls, GaussianProcessEmulator)
            and model_cls not in self._gaussian_process_emulators
        ):
            self._gaussian_process_emulators.append(model_cls)

        # Check if it's a PyTorch emulator (subclass of nn.Module) + not in PyTorch list
        if (
            issubclass(model_cls, nn.Module)
            and model_cls not in self._pytorch_emulators
        ):
            self._pytorch_emulators.append(model_cls)
        # Check it's not in non-PyTorch list
        elif model_cls not in self._non_pytorch_emulators:
            self._non_pytorch_emulators.append(model_cls)

        return model_cls

    @property
    def gaussian_process_emulators(self) -> list[type[Emulator]]:
        """Return the list of Gaussian Process emulators."""
        return self._gaussian_process_emulators

    @property
    def pytorch_emulators(self) -> list[type[Emulator]]:
        """Return the list of PyTorch-based emulators."""
        return self._pytorch_emulators

    @property
    def all_emulators(self) -> list[type[Emulator]]:
        """Return the list of all registered emulators."""
        return self._all_emulators

    @property
    def non_pytorch_emulators(self) -> list[type[Emulator]]:
        """Return the list of non-PyTorch emulators."""
        return self._non_pytorch_emulators

    @property
    def default_emulators(self) -> list[type[Emulator]]:
        """Return the list of default emulators."""
        return self._default_emulators

    def get_emulator_class(self, name: str) -> type[Emulator]:
        """
        Get the emulator class by name.

        Parameters
        ----------
        name: str
            The name of the emulator class.

        Returns
        -------
        type[Emulator]
            The emulator class if found.

        Raises
        ------
        ValueError
            If the emulator name is not found.
        """
        emulator_cls = self._emulator_registry.get(
            name.lower()
        ) or self._emulator_registry_short_name.get(name.lower())

        if emulator_cls is None:
            raise ValueError(
                f"Unknown emulator name: {name}."
                f"Available: {list(self._emulator_registry.keys())}"
            )

        return emulator_cls


# Create a default registry instance for backward compatibility
_default_registry = Registry()

# Module-level constants for backward compatibility
DEFAULT_EMULATORS = _default_registry._default_emulators
NON_PYTORCH_EMULATORS = _default_registry._non_pytorch_emulators
ALL_EMULATORS = _default_registry._all_emulators
PYTORCH_EMULATORS = _default_registry._pytorch_emulators
GAUSSIAN_PROCESS_EMULATORS = _default_registry._gaussian_process_emulators
EMULATOR_REGISTRY = _default_registry._emulator_registry
EMULATOR_REGISTRY_SHORT_NAME = _default_registry._emulator_registry_short_name


def get_emulator_class(name: str) -> type[Emulator]:
    """
    Get the emulator class by name using the default registry.

    Parameters
    ----------
    name: str
        The name of the emulator class.

    Returns
    -------
    type[Emulator]
        The emulator class if found.
    """
    return _default_registry.get_emulator_class(name)


# Overload signatures for type checking
@overload
def register(model_cls: type[Emulator]) -> type[Emulator]: ...


@overload
def register(
    *, overwrite: bool = True
) -> Callable[[type[Emulator]], type[Emulator]]: ...


# Actual implementation
def register(
    model_cls: type[Emulator] | None = None, *, overwrite: bool = True
) -> type[Emulator] | Callable[[type[Emulator]], type[Emulator]]:
    """
    Register a new emulator model to the default registry.

    Can be used as a function, a decorator without arguments, or a decorator with
    arguments.


    Parameters
    ----------
    model_cls : type[Emulator] | None
        The emulator class to register. If None, returns a decorator function.
    overwrite : bool
        If True, allows overwriting an existing model with the same name. If False,
        raises an error if a model with the same name already exists. Defaults to True.

    Returns
    -------
    type[Emulator] | Callable[[type[Emulator]], type[Emulator]]
        The registered emulator class (unchanged) or a decorator function.

    Raises
    ------
    ValueError
        If overwrite is False and a model with the same name already exists.

    """

    def decorator(cls: type[Emulator]) -> type[Emulator]:
        return _default_registry.register_model(cls, overwrite=overwrite)

    if model_cls is None:
        # Called as @register(overwrite=...) or @register()
        return decorator
    # Called as @register or register(MyClass)
    return decorator(model_cls)


__all__ = [
    "MLP",
    "EnsembleMLP",
    "EnsembleMLPDropout",
    "GaussianProcessCorrelatedMatern32",
    "GaussianProcessCorrelatedRBF",
    "GaussianProcessMatern32",
    "GaussianProcessRBF",
    "LightGBM",
    "PolynomialRegression",
    "RadialBasisFunctions",
    "RandomForest",
    "Registry",
    "SupportVectorMachine",
    "TransformedEmulator",
    "get_emulator_class",
    "register",
]
