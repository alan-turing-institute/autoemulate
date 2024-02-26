import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoemulate.emulators import MODEL_REGISTRY
from autoemulate.model_processing import _check_model_names
from autoemulate.model_processing import _get_model_name_dict
from autoemulate.model_processing import _get_models
from autoemulate.model_processing import _turn_models_into_multioutput
from autoemulate.model_processing import _wrap_models_in_pipeline


# -----------------------test getting models-------------------#
def test_get_models():
    models = _get_models(MODEL_REGISTRY)
    assert isinstance(models, list)
    model_names = [type(model).__name__ for model in models]
    assert all([model_name in MODEL_REGISTRY for model_name in model_names])


def test_check_model_names():
    models = _get_models(MODEL_REGISTRY)
    model_names = [type(model).__name__ for model in models]
    with pytest.raises(ValueError):
        _check_model_names(["NotInRegistry"], MODEL_REGISTRY)
    _check_model_names(model_names, MODEL_REGISTRY)


def test_get_models_subset():
    models = _get_models(
        MODEL_REGISTRY, model_subset=["GaussianProcessSk", "RandomForest"]
    )
    assert len(models) == 2
    model_names = [type(model).__name__ for model in models]
    assert all([model_name in MODEL_REGISTRY for model_name in model_names])


# -----------------------test get model name dict from model registry-------------------#``
def test_get_model_name_dict():
    model_name_dict = _get_model_name_dict(MODEL_REGISTRY)
    assert isinstance(model_name_dict, dict)
    assert all([model_name in model_name_dict for model_name in MODEL_REGISTRY])


# -----------------------test turning models into multioutput-------------------#
def test_turn_models_into_multioutput():
    models = _get_models(MODEL_REGISTRY)
    y = np.array([[1, 2], [3, 4]])
    models = _turn_models_into_multioutput(models, y)
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
def test_wrap_models_in_pipeline_no_scaler():
    models = _get_models(MODEL_REGISTRY)
    models = _wrap_models_in_pipeline(
        models,
        scale=False,
        scaler=StandardScaler(),
        reduce_dim=False,
        dim_reducer=PCA(),
    )
    assert isinstance(models, list)
    assert all([isinstance(model, Pipeline) for model in models])
    # assert that pipeline does have a scaler as first step
    assert all([model.steps[0][0] != "scaler" for model in models])


def test_wrap_models_in_pipeline_scaler():
    models = _get_models(MODEL_REGISTRY)
    models = _wrap_models_in_pipeline(
        models, scale=True, scaler=StandardScaler(), reduce_dim=False, dim_reducer=PCA()
    )
    assert isinstance(models, list)
    assert all([isinstance(model, Pipeline) for model in models])
    # assert that pipeline does have a scaler as first step
    assert all([model.steps[0][0] == "scaler" for model in models])


def test_wrap_models_in_pipeline_no_scaler_dim_reducer():
    models = _get_models(MODEL_REGISTRY)
    models = _wrap_models_in_pipeline(
        models, scale=False, scaler=StandardScaler(), reduce_dim=True, dim_reducer=PCA()
    )
    assert isinstance(models, list)
    # check that dim_reducer is first step
    assert all([isinstance(model, Pipeline) for model in models])
    assert all([model.steps[0][0] == "dim_reducer" for model in models])


def test_wrap_models_in_pipeline_scaler_dim_reducer():
    models = _get_models(MODEL_REGISTRY)
    models = _wrap_models_in_pipeline(
        models, scale=True, scaler=StandardScaler(), reduce_dim=True, dim_reducer=PCA()
    )
    assert isinstance(models, list)
    assert all([isinstance(model, Pipeline) for model in models])
    # assert that pipeline does have a scaler as first step
    assert all([model.steps[0][0] == "scaler" for model in models])
    assert all([model.steps[1][0] == "dim_reducer" for model in models])
