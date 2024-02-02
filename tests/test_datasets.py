import pytest

from autoemulate.datasets import fetch_data


def test_valid_dataset_name():
    for name in [
        "cardiac1",
        "cardiac2",
        "cardiac3",
        "cardiac4",
        "cardiac5",
        "cardiac6",
        "climate1",
        "engineering1",
    ]:
        X, y = fetch_data(name)
        assert X is not None
        assert y is not None


def test_train_test_split():
    X_train, X_test, y_train, y_test = fetch_data("cardiac1", split=True)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
