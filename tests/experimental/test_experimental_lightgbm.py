import numpy as np
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


def test_lgbm_deterministic_with_seed():
    # Generate very small, highly noisy data to maximize stochasticity effects
    rng = np.random.RandomState(0)
    x = torch.tensor(rng.normal(size=(8, 2)), dtype=torch.float32)
    y = torch.tensor(rng.normal(size=(8,)), dtype=torch.float32)
    # Use some of the training points as test points to amplify differences
    x2 = x[:4].clone()

    # Use parameters that introduce stochasticity
    params = {
        "n_estimators": 20,
        "subsample": 0.5,
        "colsample_bytree": 0.5,
        "random_seed": 42,
        "num_leaves": 31,
        "max_depth": 5,
        "learning_rate": 0.1,
        "boosting_type": "dart",
    }

    model1 = LightGBM(**params)
    model2 = LightGBM(**params)
    model1.fit(x, y)
    model2.fit(x, y)

    # Change the seed for model3
    params["random_seed"] = 24
    model3 = LightGBM(**params)
    model3.fit(x, y)

    pred1 = model1.predict(x2)
    pred2 = model2.predict(x2)
    assert isinstance(pred1, TensorLike)
    assert isinstance(pred2, TensorLike)
    assert torch.allclose(pred1, pred2)

    assert torch.equal(pred1, pred2)

    # TODO: find a way to get LGBM to produce different outputs with different seeds
    # pred3 = model3.predict(x2)
    # assert isinstance(pred3, TensorLike)
    # assert not torch.equal(pred1, pred3)
