import pytest

from autoemulate.emulators import ConditionalNeuralProcess
from autoemulate.emulators import GaussianProcessSklearn
from autoemulate.emulators import RadialBasisFunctions
from autoemulate.model_registry import ModelRegistry


@pytest.fixture
def model_registry():
    model_registry = ModelRegistry()
    model_registry.register_model(
        "RadialBasisFunctions", RadialBasisFunctions, is_core=True
    )
    model_registry.register_model(
        "GaussianProcessSklearn", GaussianProcessSklearn, is_core=False
    )
    model_registry.register_model(
        "ConditionalNeuralProcess", ConditionalNeuralProcess, is_core=True
    )
    return model_registry


def test_init():
    model_registry = ModelRegistry()
    assert model_registry.models == {}
    assert model_registry.core_model_names == []


def test_register_model(model_registry):
    assert model_registry.models["RadialBasisFunctions"] == RadialBasisFunctions
    assert model_registry.core_model_names == [
        "RadialBasisFunctions",
        "ConditionalNeuralProcess",
    ]


def test_get_core_models(model_registry):
    core_models = model_registry.get_core_models()
    assert isinstance(core_models, list)
    assert len(core_models) == 2
    assert core_models[0].__class__ == RadialBasisFunctions


def test_get_all_models(model_registry):
    all_models = model_registry.get_all_models()
    assert isinstance(all_models, list)
    assert len(all_models) == 3
    assert all_models[1].__class__ == GaussianProcessSklearn


# check get_model_names
def test_get_model_names_all(model_registry):
    model_names = model_registry.get_model_names()
    assert isinstance(model_names, dict)
    assert len(model_names) == 3
    assert model_names["ConditionalNeuralProcess"] == "cnp"


def test_get_model_names_subset(model_registry):
    model_names = model_registry.get_model_names(model_subset=["cnp", "gps"])
    assert isinstance(model_names, dict)
    assert len(model_names) == 2


def test_get_model_names_mix(model_registry):
    model_names = model_registry.get_model_names(
        model_subset=["ConditionalNeuralProcess", "rbf"]
    )
    assert isinstance(model_names, dict)
    assert len(model_names) == 2
    assert model_names["ConditionalNeuralProcess"] == "cnp"
    assert model_names["RadialBasisFunctions"] == "rbf"


def test_get_model_names_str(model_registry):
    model_names = model_registry.get_model_names(model_subset="cnp")
    assert isinstance(model_names, dict)
    assert len(model_names) == 1
    assert model_names["ConditionalNeuralProcess"] == "cnp"


def test_get_model_names_invalid(model_registry):
    with pytest.raises(ValueError):
        model_registry.get_model_names(model_subset=["cnp", "gps", "rbf", "invalid"])


def test_get_model_names_invalid_str(model_registry):
    with pytest.raises(ValueError):
        model_registry.get_model_names(model_subset="invalid")


# check get_models -------------------------------------------
def test_get_models(model_registry):
    models = model_registry.get_models()
    # default is core models
    assert isinstance(models, list)
    assert len(models) == 2
    assert models[0].__class__ == RadialBasisFunctions
    assert models[1].__class__ == ConditionalNeuralProcess


def test_get_models_subset(model_registry):
    models = model_registry.get_models(model_subset=["cnp", "gps"])
    assert isinstance(models, list)
    assert len(models) == 2
    assert models[0].__class__ == GaussianProcessSklearn
    assert models[1].__class__ == ConditionalNeuralProcess
