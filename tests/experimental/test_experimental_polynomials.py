from autoemulate.experimental.emulators.polynomials import (
    SecondOrderPolynomial,
)
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import TensorLike


def test_predict_sop(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    sop = SecondOrderPolynomial(x, y)
    sop.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = sop.predict(x2)
    assert isinstance(y_pred, TensorLike)


def test_predict_sop_2d(sample_data_y2d, new_data_y2d):
    x, y = sample_data_y2d
    sop = SecondOrderPolynomial(x, y)
    sop.fit(x, y)
    x2, _ = new_data_y2d
    y_pred = sop.predict(x2)
    assert isinstance(y_pred, TensorLike)
    assert y_pred.shape == (20, 2)


def test_tune_sop(sample_data_y1d):
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5)
    scores, configs = tuner.run(SecondOrderPolynomial)
    assert len(scores) == 5
    assert len(configs) == 5


# TODO: add determinism test after merging #512
