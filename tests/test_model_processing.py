import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoemulate.emulators import GaussianProcess
from autoemulate.emulators import RandomForest
from autoemulate.emulators import RBF
from autoemulate.model_processing import _check_model_names
from autoemulate.model_processing import _get_models
from autoemulate.model_processing import _turn_models_into_multioutput
from autoemulate.model_processing import _wrap_models_in_pipeline
from autoemulate.utils import get_model_name


@pytest.fixture()
def model_registry():
    return {
        "GaussianProcess": GaussianProcess(),
        "RandomForest": RandomForest(),
        "RadialBasisFunctions": RBF(),
    }


# -----------------------test getting models-------------------#
def test_check_model_names(model_registry):
    model_names = model_registry.keys()
    with pytest.raises(ValueError):
        _check_model_names(["NotInRegistry"], model_registry)
    _check_model_names(model_names, model_registry)


def test_get_models(model_registry):
    models = _get_models(model_registry)
    assert isinstance(models, list)
    model_names = [get_model_name(model) for model in models]
    assert all([model_name in model_registry for model_name in model_names])
    # check all values are scikit-learn estimators
    assert all([isinstance(model, BaseEstimator) for model in models])


def test_get_models_subset(model_registry):
    models = _get_models(
        model_registry, model_subset=["GaussianProcess", "RandomForest"]
    )
    assert len(models) == 2
    model_names = [get_model_name(model) for model in models]
    assert all([model_name in model_registry for model_name in model_names])
    assert all([isinstance(model, BaseEstimator) for model in models])


# -----------------------test turning models into multioutput-------------------#
def test_turn_models_into_multioutput(model_registry):
    models = _get_models(model_registry)
    y = np.array([[1, 2], [3, 4]])
    models = _turn_models_into_multioutput(models, y)
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
    models = _get_models(model_registry)
    models = _wrap_models_in_pipeline(
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
    models = _get_models(model_registry)
    models = _wrap_models_in_pipeline(
        models, scale=True, scaler=StandardScaler(), reduce_dim=False, dim_reducer=PCA()
    )
    assert isinstance(models, list)
    assert all([isinstance(model, Pipeline) for model in models])
    # assert that pipeline does have a scaler as first step
    assert all([model.steps[0][0] == "scaler" for model in models])


def test_wrap_models_in_pipeline_no_scaler_dim_reducer(model_registry):
    models = _get_models(model_registry)
    models = _wrap_models_in_pipeline(
        models, scale=False, scaler=StandardScaler(), reduce_dim=True, dim_reducer=PCA()
    )
    assert isinstance(models, list)
    # check that dim_reducer is first step
    assert all([isinstance(model, Pipeline) for model in models])
    assert all([model.steps[0][0] == "dim_reducer" for model in models])


def test_wrap_models_in_pipeline_scaler_dim_reducer(model_registry):
    models = _get_models(model_registry)
    models = _wrap_models_in_pipeline(
        models, scale=True, scaler=StandardScaler(), reduce_dim=True, dim_reducer=PCA()
    )
    assert isinstance(models, list)
    assert all([isinstance(model, Pipeline) for model in models])
    # assert that pipeline does have a scaler as first step
    assert all([model.steps[0][0] == "scaler" for model in models])
    assert all([model.steps[1][0] == "dim_reducer" for model in models])
