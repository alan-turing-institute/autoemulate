import torch
from autoemulate.experimental.emulators.random_forest import (
    RandomForest,
)
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import TensorLike


def test_predict_rf(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    rf = RandomForest(x, y)
    rf.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = rf.predict(x2)
    assert isinstance(y_pred, TensorLike)


def test_predict_rf_2d(sample_data_y2d, new_data_y2d):
    x, y = sample_data_y2d
    rf = RandomForest(x, y)
    rf.fit(x, y)
    x2, _ = new_data_y2d
    y_pred = rf.predict(x2)
    assert isinstance(y_pred, TensorLike)
    assert y_pred.shape == (20, 2)


def test_tune_rf(sample_data_y1d):
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5)
    scores, configs = tuner.run(RandomForest)
    assert len(scores) == 5
    assert len(configs) == 5


def test_rf_deterministic_with_seed(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    x2, _ = new_data_y1d
    model1 = RandomForest(x, y, random_seed=42)
    model2 = RandomForest(x, y, random_seed=42)
    model1.fit(x, y)
    model2.fit(x, y)
    pred1 = model1.predict(x2)
    pred2 = model2.predict(x2)
    assert isinstance(pred1, TensorLike)
    assert isinstance(pred2, TensorLike)
    assert torch.allclose(pred1, pred2)
