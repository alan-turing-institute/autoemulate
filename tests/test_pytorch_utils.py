import pytest
import torch
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from autoemulate.emulators import GaussianProcess
from autoemulate.emulators import RandomForest
from autoemulate.utils import extract_pytorch_model


@pytest.fixture
def pytorch_model():
    return GaussianProcess()


@pytest.fixture
def non_pytorch_model():
    return RandomForest()


@pytest.fixture
def non_pytorch_multiout_model():
    return MultiOutputRegressor(SVR())


@pytest.fixture
def Xy():
    X, y = make_regression(n_samples=100, n_features=3, n_targets=2)
    return X, y


# test_error_when_not_fitted
def test_error_when_not_fitted(pytorch_model):
    with pytest.raises(ValueError):
        extract_pytorch_model(pytorch_model)


# test standalone model
def test_extract_when_fitted(pytorch_model, Xy):
    pytorch_model.fit(*Xy)
    model = extract_pytorch_model(pytorch_model)
    assert isinstance(model, torch.nn.Module)


def test_error_when_not_pytorch_model(non_pytorch_model, Xy):
    non_pytorch_model.fit(*Xy)
    with pytest.raises(ValueError):
        extract_pytorch_model(non_pytorch_model)


def test_error_when_multiout_model(non_pytorch_multiout_model, Xy):
    non_pytorch_multiout_model.fit(*Xy)
    with pytest.raises(ValueError):
        extract_pytorch_model(non_pytorch_multiout_model)


# test pipeline
def test_extract_when_fitted_pipeline(pytorch_model, Xy):
    pytorch_model = Pipeline([("model", pytorch_model)])
    pytorch_model.fit(*Xy)
    model = extract_pytorch_model(pytorch_model)
    assert isinstance(model, torch.nn.Module)


def test_error_when_non_pytorch_pipeline(non_pytorch_model, Xy):
    non_pytorch_model = Pipeline([("model", non_pytorch_model)])
    non_pytorch_model.fit(*Xy)
    with pytest.raises(ValueError):
        extract_pytorch_model(non_pytorch_model)


def test_error_when_multiout_pipeline(non_pytorch_multiout_model, Xy):
    non_pytorch_multiout_model = Pipeline([("model", non_pytorch_multiout_model)])
    non_pytorch_multiout_model.fit(*Xy)
    with pytest.raises(ValueError):
        extract_pytorch_model(non_pytorch_multiout_model)


def test_warning_when_scaled_or_reduced(pytorch_model, Xy, capsys):
    pytorch_model = Pipeline([("scaler", StandardScaler()), ("model", pytorch_model)])
    pytorch_model.fit(*Xy)
    extract_pytorch_model(pytorch_model)
    captured = capsys.readouterr()
    assert (
        "Warning: Data preprocessing is not included in the extracted model"
        in captured.out
    )
