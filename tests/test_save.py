import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from autoemulate.core.logging_config import get_configured_logger
from autoemulate.core.results import Result  # , Results
from autoemulate.core.save import ModelSerialiser
from autoemulate.emulators.base import Emulator
from autoemulate.emulators.gaussian_process.exact import (
    GaussianProcess,
)
from autoemulate.emulators.polynomials import PolynomialRegression
from autoemulate.emulators.random_forest import RandomForest
from autoemulate.emulators.transformed.base import TransformedEmulator
from autoemulate.transforms import StandardizeTransform

logger, _ = get_configured_logger("info")


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
            model_serialiser._save_model(model, None, None)
            model_filename = model_serialiser._get_model_filename(model)
            expected_path = Path(model_filename)
            assert expected_path.exists()
        finally:
            os.chdir(original_wd)


def test_save_model_w_name(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        original_wd = os.getcwd()
        os.chdir(temp_dir)

        try:
            model_serialiser._save_model(model, "rf")
            expected_path = Path("rf.joblib")
            assert expected_path.exists()
        finally:
            os.chdir(original_wd)


def test_save_model_w_dir(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir)
        model_serialiser._save_model(model, None, test_path)
        assert test_path.exists()
        assert os.path.exists(
            os.path.join(test_path, model_serialiser._get_model_filename(model))
        )


def test_load_model(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        original_wd = os.getcwd()
        os.chdir(temp_dir)
        try:
            model_serialiser._save_model(model)
            loaded_model = model_serialiser._load_model("RandomForest.joblib")
            assert isinstance(loaded_model, type(model))
        finally:
            os.chdir(original_wd)


def test_invalid_file_path(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        invalid_path = os.path.join(temp_dir, "/invalid/path/model")
        with pytest.raises(Exception):  # noqa: B017, PT011
            # only the / makes it invalid
            model_serialiser._save_model(model, None, invalid_path)
        with pytest.raises(Exception):  # noqa: B017, PT011
            model_serialiser._load_model(invalid_path)


def test_save_model_with_model_name(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        original_wd = os.getcwd()
        os.chdir(temp_dir)
        try:
            model_name = "custom_name"
            saved_path = model_serialiser._save_model(model, model_name)
            assert Path(saved_path).exists()
            assert Path(saved_path).name == model_name + ".joblib"
        finally:
            os.chdir(original_wd)


def test_save_and_load_result(model_serialiser, sample_data_y2d):
    x, y = sample_data_y2d
    em = TransformedEmulator(
        x,
        y,
        x_transforms=[StandardizeTransform()],
        y_transforms=None,
        model=GaussianProcess,
    )
    params = GaussianProcess.get_random_params()
    result = Result(
        id=12345,
        model_name="dummy_model",
        model=em,
        params=params,
        r2_test=0.9,
        r2_test_std=0.01,
        r2_train=0.95,
        r2_train_std=0.015,
        rmse_test=0.1,
        rmse_train=0.05,
        rmse_test_std=0.02,
        rmse_train_std=0.025,
    )
    with TemporaryDirectory() as temp_dir:
        original_wd = os.getcwd()
        os.chdir(temp_dir)
        try:
            path = Path(temp_dir)
            filename = f"{result.model_name}_{result.id}"
            result_path = model_serialiser._save_result(result, filename, path)

            expected_metadata_path = os.path.join(
                path, "dummy_model_12345_metadata.csv"
            )
            expected_model_filename_path = os.path.join(
                path, "dummy_model_12345.joblib"
            )

            assert os.path.exists(expected_metadata_path)
            assert os.path.exists(expected_model_filename_path)

            loaded = model_serialiser._load_result(result_path)

            assert isinstance(loaded, Result)
            assert loaded.id == result.id
            assert loaded.model_name == result.model_name

            for k, v in result.params.items():
                loaded_v = loaded.params[k]
                if callable(v):
                    assert str(loaded_v) in str(v)
                else:
                    assert loaded_v == v
            assert loaded.r2_test == result.r2_test
            assert loaded.rmse_test == result.rmse_test

        finally:
            os.chdir(original_wd)


def test_load_result_no_metadata(model_serialiser, model):
    with TemporaryDirectory() as temp_dir:
        original_wd = os.getcwd()
        os.chdir(temp_dir)
        try:
            model_path = model_serialiser._save_model(model)
            loaded = model_serialiser._load_result(model_path)
            assert isinstance(loaded, Emulator)
        finally:
            os.chdir(original_wd)
