import pytest
from autoemulate.save import ModelSerialiser
from sklearn.ensemble import RandomForestRegressor
import os


@pytest.fixture
def model_serialiser():
    return ModelSerialiser()


@pytest.fixture
def model():
    return RandomForestRegressor()


def test_save_model(model_serialiser, model):
    path = "test_model.joblib"
    model_serialiser.save_model(model, path)
    assert os.path.exists(path)
    os.remove(path)


def test_load_model(model_serialiser, model):
    path = "test_model.joblib"
    model_serialiser.save_model(model, path)
    loaded_model = model_serialiser.load_model(path)
    assert isinstance(loaded_model, type(model))
    os.remove(path)
