import pytest
import torch
from autoemulate.core.tuner import Tuner
from autoemulate.core.types import TensorLike
from autoemulate.emulators.random_forest import RandomForest


def test_predict_rf(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    rf = RandomForest(x, y)
    rf.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = rf.predict(x2)
    assert isinstance(y_pred, TensorLike)
    assert not y_pred.requires_grad

    with pytest.raises(ValueError, match="Gradient calculation is not supported."):
        rf.predict(x2, with_grad=True)


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
    scores, params_list = tuner.run(RandomForest)
    assert len(scores) == 5
    assert len(params_list) == 5


def test_rf_deterministic_with_seed(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    x2, _ = new_data_y1d
    seed = 42
    model1 = RandomForest(x, y, random_seed=seed)
    model2 = RandomForest(x, y, random_seed=seed)
    model1.fit(x, y)
    model2.fit(x, y)
    pred1 = model1.predict(x2)
    pred2 = model2.predict(x2)
    assert isinstance(pred1, TensorLike)
    assert isinstance(pred2, TensorLike)
    assert torch.allclose(pred1, pred2)
    # Change the seed for model3
    new_seed = 24
    model3 = RandomForest(x, y, random_seed=new_seed)
    model3.fit(x, y)
    pred3 = model3.predict(x2)
    assert isinstance(pred3, TensorLike)
    assert not torch.allclose(pred1, pred3)
