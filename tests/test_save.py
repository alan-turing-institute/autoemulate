import json
import os
import shutil

import pytest

from autoemulate.emulators import GaussianProcessSk
from autoemulate.emulators import RandomForest
from autoemulate.save import ModelSerialiser
from autoemulate.utils import get_model_name


@pytest.fixture
def model_serialiser():
    return ModelSerialiser()


@pytest.fixture
def model():
    return RandomForest()


@pytest.fixture
def models():
    return {"RandomForest": RandomForest(), "GaussianProcesses": GaussianProcessSk()}


@pytest.fixture
def test_path():
    return "test_model"


def test_save_model_w_path(model_serialiser, model, test_path):
    model_serialiser._save_model(model, test_path)
    assert os.path.exists(test_path)
    assert os.path.exists(model_serialiser._get_meta_path(test_path))

    with open(model_serialiser._get_meta_path(test_path), "r") as f:
        meta = json.load(f)
    assert "model" in meta
    assert "scikit-learn" in meta
    assert "numpy" in meta

    os.remove(test_path)
    os.remove(model_serialiser._get_meta_path(test_path))


def test_save_model_w_dir(model_serialiser, model):
    test_dir = "test_dir"
    os.makedirs(test_dir, exist_ok=True)
    model_serialiser._save_model(model, test_dir)
    model_name = get_model_name(model)
    assert os.path.exists(os.path.join(test_dir, model_name))

    os.remove(os.path.join(test_dir, model_name))
    os.remove(os.path.join(test_dir, model_serialiser._get_meta_path(model_name)))
    os.rmdir(test_dir)


def test_save_model_wo_path(model_serialiser, model):
    model_serialiser._save_model(model, None)
    model_name = get_model_name(model)
    assert os.path.exists(model_name)
    assert os.path.exists(model_serialiser._get_meta_path(model_name))

    with open(model_serialiser._get_meta_path(model_name), "r") as f:
        meta = json.load(f)
    assert "model" in meta
    assert "scikit-learn" in meta
    assert "numpy" in meta

    os.remove(model_name)
    os.remove(model_serialiser._get_meta_path(model_name))


def test_save_models_wo_dir(model_serialiser, models):
    model_serialiser._save_models(models, None)
    model_names = models.keys()
    assert all([os.path.exists(model_name) for model_name in model_names])
    assert all(
        [
            os.path.exists(model_serialiser._get_meta_path(model_name))
            for model_name in model_names
        ]
    )

    for model_name in model_names:
        os.remove(model_name)
        os.remove(model_serialiser._get_meta_path(model_name))


def test_save_models_w_dir(model_serialiser, models):
    test_dir = "test_dir"
    os.makedirs(test_dir, exist_ok=True)
    model_serialiser._save_models(models, test_dir)
    model_names = models.keys()
    assert all(
        [
            os.path.exists(os.path.join(test_dir, model_name))
            for model_name in model_names
        ]
    )
    assert all(
        [
            os.path.exists(
                os.path.join(test_dir, model_serialiser._get_meta_path(model_name))
            )
            for model_name in model_names
        ]
    )

    for model_name in model_names:
        os.remove(os.path.join(test_dir, model_name))
        os.remove(os.path.join(test_dir, model_serialiser._get_meta_path(model_name)))
    os.rmdir(test_dir)


def test_load_model(model_serialiser, model, test_path):
    model_serialiser._save_model(model, test_path)
    loaded_model = model_serialiser._load_model(test_path)
    assert isinstance(loaded_model, type(model))
    os.remove(test_path)
    os.remove(model_serialiser._get_meta_path(test_path))


def test_load_model_with_missing_meta_file(model_serialiser, model, test_path):
    model_serialiser._save_model(model, test_path)
    os.remove(model_serialiser._get_meta_path(test_path))
    with pytest.raises(FileNotFoundError):
        model_serialiser._load_model(test_path)
    os.remove(test_path)


def test_invalid_file_path(model_serialiser, model):
    with pytest.raises(Exception):
        model_serialiser._save_model(model, "/invalid/path/model")
    with pytest.raises(Exception):
        model_serialiser._load_model("/invalid/path/model")
