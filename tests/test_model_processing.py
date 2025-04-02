import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoemulate.emulators import GaussianProcessSklearn
from autoemulate.emulators import RadialBasisFunctions
from autoemulate.emulators import RandomForest
from autoemulate.emulators import SupportVectorMachines
from autoemulate.model_processing import ModelPrepPipeline
from autoemulate.model_registry import ModelRegistry
from autoemulate.utils import get_model_name


@pytest.fixture
def model_registry():
    model_registry = ModelRegistry()
    model_registry.register_model(
        "RadialBasisFunctions", RadialBasisFunctions, is_core=True
    )
    model_registry.register_model(
        "GaussianProcessSklearn", GaussianProcessSklearn, is_core=True
    )
    model_registry.register_model(
        "SupportVectorMachines", SupportVectorMachines, is_core=True
    )
    return model_registry


# -----------------------test turning models into multioutput-------------------#
def test_turn_models_into_multioutput(model_registry):
    models = model_registry.get_models()
    y = np.array([[1, 2], [3, 4]])
    models = ModelPrepPipeline._turn_models_into_multioutput(models, y)
    print(f"models === {models}")
    assert isinstance(models, list)
    # check that non-native multioutput models are wrapped in MultiOutputRegressor
    assert all(
        [
            isinstance(model, MultiOutputRegressor)
            for model in models
            if not model._more_tags().get("multioutput")
        ]
    )


# -----------------------test wrapping models in pipeline-------------------#
def test_wrap_models_in_pipeline_no_scaler(model_registry):
    models = model_registry.get_models()
    models = ModelPrepPipeline._wrap_models_in_pipeline(
        models,
        scale=False,
        scaler=StandardScaler(),
        reduce_dim=False,
        dim_reducer=PCA(),
    )
    assert isinstance(models, list)
    assert all([isinstance(model, Pipeline) for model in models])
    # assert that pipeline does have a scaler
    assert all([model.steps[0][0] != "scaler" for model in models])


def test_wrap_models_in_pipeline_scaler(model_registry):
    models = model_registry.get_models()
    models = ModelPrepPipeline._wrap_models_in_pipeline(
        models, scale=True, scaler=StandardScaler(), reduce_dim=False, dim_reducer=PCA()
    )
    assert isinstance(models, list)
    assert all([isinstance(model, Pipeline) for model in models])
    # assert that pipeline does have a scaler as first step
    assert all([model.steps[0][0] == "scaler" for model in models])


def test_wrap_models_in_pipeline_no_scaler_dim_reducer(model_registry):
    models = model_registry.get_models()
    models = ModelPrepPipeline._wrap_models_in_pipeline(
        models, scale=False, scaler=StandardScaler(), reduce_dim=True, dim_reducer=PCA()
    )
    assert isinstance(models, list)
    # check that dim_reducer is first step
    assert all([isinstance(model, Pipeline) for model in models])
    assert all([model.steps[0][0] == "dim_reducer" for model in models])


def test_wrap_models_in_pipeline_scaler_dim_reducer(model_registry):
    models = model_registry.get_models()
    models = ModelPrepPipeline._wrap_models_in_pipeline(
        models, scale=True, scaler=StandardScaler(), reduce_dim=True, dim_reducer=PCA()
    )
    assert isinstance(models, list)
    assert all([isinstance(model, Pipeline) for model in models])
    # assert that pipeline does have a scaler as first step
    assert all([model.steps[0][0] == "scaler" for model in models])
    assert all([model.steps[1][0] == "dim_reducer" for model in models])
