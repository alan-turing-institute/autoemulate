import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from autoemulate.experimental.emulators.polynomials import PolynomialRegression
from autoemulate.experimental.emulators.random_forest import RandomForest
from autoemulate.save import ModelSerialiser

logger = logging.getLogger(__name__)


@pytest.fixture
def model_serialiser():
    return ModelSerialiser(logger=logger)


@pytest.fixture
def model(sample_data_y2d):
    x, y = sample_data_y2d
    return RandomForest(x, y)


@pytest.fixture
def models(sample_data_y2d):
    x, y = sample_data_y2d
    return [RandomForest(x, y), PolynomialRegression(x, y)]


@pytest.fixture
def test_path():
    return "test_model"


def test_save_model_wo_path(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        original_wd = os.getcwd()
        os.chdir(temp_dir)

        try:
            model_serialiser._save_model(model, None)
            model_name = model_serialiser._get_model_name(model)
            expected_path = Path(model_name)
            assert expected_path.exists()
        finally:
            os.chdir(original_wd)


def test_save_model_w_name(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        original_wd = os.getcwd()
        os.chdir(temp_dir)

        try:
            model_serialiser._save_model(model, "rf")
            expected_path = Path("rf")
            assert expected_path.exists()
        finally:
            os.chdir(original_wd)


def test_save_model_w_dir(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir)
        model_serialiser._save_model(model, test_path)
        assert test_path.exists()
        assert (test_path / model_serialiser._get_model_name(model)).exists()


def test_load_model(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "test_model"
        model_serialiser._save_model(model, test_path)
        loaded_model = model_serialiser._load_model(test_path)
        assert isinstance(loaded_model, type(model))


def test_invalid_file_path(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        invalid_path = Path(temp_dir) / "/invalid/path/model"
        with pytest.raises(FileNotFoundError):
            # only the / makes it invalid
            model_serialiser._save_model(model, invalid_path)
        with pytest.raises(FileNotFoundError):
            model_serialiser._load_model(invalid_path)
