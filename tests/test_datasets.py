import pytest

from autoemulate.datasets import fetch_cardiac_data


def test_valid_dataset_name():
    for name in ["atrial_cell", "four_chamber", "circ_adapt"]:
        X, y = fetch_cardiac_data(name)
        assert X is not None
        assert y is not None


def test_train_test_split():
    X_train, X_test, y_train, y_test = fetch_cardiac_data(
        "atrial_cell", train_test=True
    )
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_test_size():
    for size in [0.1, 0.5]:
        X_train, X_test, _, _ = fetch_cardiac_data(
            "atrial_cell", train_test=True, test_size=size
        )
        assert len(X_test) / (len(X_train) + len(X_test)) == pytest.approx(size)
