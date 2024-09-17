import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from autoemulate.emulators import GaussianProcessSklearn
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
    return [RandomForest(), GaussianProcessSklearn()]


@pytest.fixture
def test_path():
    return "test_model"


def test_save_model_wo_path(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        original_wd = os.getcwd()
        os.chdir(temp_dir)

        try:
            model_serialiser._save_model(model, None)
            model_name = get_model_name(model)
            expected_path = Path(model_name)
            expected_meta_path = model_serialiser._get_meta_path(model_name)
            assert expected_path.exists()
            assert expected_meta_path.exists()
        finally:
            os.chdir(original_wd)


def test_save_model_w_name(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "test_model"
        model_serialiser._save_model(model, test_path)
        meta_path = model_serialiser._get_meta_path(test_path)
        assert test_path.exists()
        assert meta_path.exists()

        with open(meta_path, "r") as f:
            meta = json.load(f)
        assert "model" in meta
        assert "scikit-learn" in meta
        assert "numpy" in meta


def test_save_model_w_dir(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir)
        model_serialiser._save_model(model, test_path)
        assert test_path.exists()
        assert (test_path / "RandomForest").exists()
        assert (test_path / "RandomForest_meta.json").exists()


def test_load_model(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "test_model"
        model_serialiser._save_model(model, test_path)
        loaded_model = model_serialiser._load_model(test_path)
        assert isinstance(loaded_model, type(model))


def test_load_model_with_missing_meta_file(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "test_model"
        model_serialiser._save_model(model, test_path)
        meta_path = model_serialiser._get_meta_path(test_path)
        meta_path.unlink()
        with pytest.raises(FileNotFoundError):
            model_serialiser._load_model(test_path)


def test_invalid_file_path(model_serialiser, model):
    with pytest.raises(Exception):
        # only the / makes it invalid
        model_serialiser._save_model(model, "/invalid/path/model")
    with pytest.raises(Exception):
        model_serialiser._load_model("/invalid/path/model")


def test_save_models_wo_path(model_serialiser, models):
    with TemporaryDirectory() as temp_dir:
        original_wd = os.getcwd()
        os.chdir(temp_dir)

        try:
            model_serialiser._save_models(models, None)
            for model in models:
                model_name = get_model_name(model)
                expected_path = Path(model_name)
                expected_meta_path = model_serialiser._get_meta_path(model_name)
                assert expected_path.exists()
                assert expected_meta_path.exists()
        finally:
            os.chdir(original_wd)
