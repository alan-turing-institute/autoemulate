from autoemulate.experimental.emulators.lightgbm import (
    LightGBM,
)
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
