import torch
from autoemulate.experimental.emulators.lightgbm import LightGBM
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import TensorLike


def test_predict_lightgbm(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    lgbm = LightGBM()
    lgbm.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = lgbm.predict(x2)
    assert isinstance(y_pred, TensorLike)


def test_tune_lightgbm(sample_data_y1d):
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5)
    scores, configs = tuner.run(LightGBM)
    assert len(scores) == 5
    assert len(configs) == 5


def test_lightgm_class_name_returned():
    lgbm = LightGBM()
    assert lgbm.model_name() == "LightGBM"


def test_lgbm_deterministic_with_seed(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    x2, _ = new_data_y1d
    model1 = LightGBM()
    model2 = LightGBM()
    model1.fit(x, y)
    model2.fit(x, y)
    pred1 = model1.predict(x2)
    pred2 = model2.predict(x2)
    assert isinstance(pred1, TensorLike)
    assert isinstance(pred2, TensorLike)
    assert torch.allclose(pred1, pred2)
