import torch
from autoemulate.experimental.emulators.polynomials import (
    PolynomialRegression,
    PolynomialRegressionOld,
)
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import TensorLike


def test_predict_sop(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    sop = PolynomialRegression(x, y)
    sop.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = sop.predict(x2)
    assert isinstance(y_pred, TensorLike)


def test_sop_sk_and_pytorch(sample_data_y2d, new_data_y2d):
    x, y = sample_data_y2d
    x2, y2 = new_data_y2d

    model1 = PolynomialRegression(x, y)
    model1.fit(x, y)
    pred1 = model1.predict(x2)

    model2 = PolynomialRegressionOld(x, y)
    model2.fit(x, y)
    pred2 = model2.predict(x2)
    print(pred1)
    print(pred2)
    print(y2)
    assert torch.allclose(pred1, pred2)  # ignore


def test_predict_sop_2d(sample_data_y2d, new_data_y2d):
    x, y = sample_data_y2d
    sop = PolynomialRegression(x, y)
    sop.fit(x, y)
    x2, _ = new_data_y2d
    y_pred = sop.predict(x2)
    assert isinstance(y_pred, TensorLike)
    assert y_pred.shape == (20, 2)


def test_tune_sop(sample_data_y1d):
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5)
    scores, configs = tuner.run(PolynomialRegression)
    assert len(scores) == 5
    assert len(configs) == 5


def test_sop_predict_deterministic_with_seed(sample_data_y2d, new_data_y2d):
    """
    Test that fitting two models with the same seed and data
    produces identical predictions.
    """
    x, y = sample_data_y2d
    x2, _ = new_data_y2d

    # Set a random seed for reproducibility
    seed = 42
    model1 = PolynomialRegression(x, y, random_seed=seed)
    model1.fit(x, y)
    pred1 = model1.predict(x2)

    # Use the same seed to ensure deterministic behavior
    model2 = PolynomialRegression(x, y, random_seed=seed)
    model2.fit(x, y)
    pred2 = model2.predict(x2)

    # Use a different seed to ensure deterministic behavior
    new_seed = 43
    model3 = PolynomialRegression(x, y, random_seed=new_seed)
    model3.fit(x, y)
    pred3 = model3.predict(x2)

    assert isinstance(pred1, torch.Tensor)
    assert isinstance(pred2, torch.Tensor)
    assert isinstance(pred3, torch.Tensor)
    assert torch.allclose(pred1, pred2)
    msg = "Predictions should differ with different seeds."
    assert not torch.allclose(pred1, pred3), msg
