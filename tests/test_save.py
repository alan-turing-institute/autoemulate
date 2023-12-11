import pytest
from autoemulate.save import ModelSerialiser
from sklearn.ensemble import RandomForestRegressor
import os
import json


@pytest.fixture
def model_serialiser():
    return ModelSerialiser()


@pytest.fixture
def model():
    return RandomForestRegressor()


@pytest.fixture
def test_path():
    return "test_model.joblib"


def test_save_model(model_serialiser, model, test_path):
    model_serialiser.save_model(model, test_path)
    assert os.path.exists(test_path)
    assert os.path.exists(model_serialiser.get_meta_path(test_path))

    with open(model_serialiser.get_meta_path(test_path), "r") as f:
        meta = json.load(f)
    assert "model" in meta
    assert "scikit-learn" in meta
    assert "numpy" in meta

    os.remove(test_path)
    os.remove(model_serialiser.get_meta_path(test_path))


def test_load_model(model_serialiser, model, test_path):
    model_serialiser.save_model(model, test_path)
    loaded_model = model_serialiser.load_model(test_path)
    assert isinstance(loaded_model, type(model))
    os.remove(test_path)
    os.remove(model_serialiser.get_meta_path(test_path))


def test_load_model_with_missing_meta_file(model_serialiser, model, test_path):
    model_serialiser.save_model(model, test_path)
    os.remove(model_serialiser.get_meta_path(test_path))
    with pytest.raises(FileNotFoundError):
        model_serialiser.load_model(test_path)
    os.remove(test_path)


def test_invalid_file_path(model_serialiser, model):
    with pytest.raises(Exception):
        model_serialiser.save_model(model, "/invalid/path/model.joblib")
    with pytest.raises(Exception):
        model_serialiser.load_model("/invalid/path/model.joblib")
