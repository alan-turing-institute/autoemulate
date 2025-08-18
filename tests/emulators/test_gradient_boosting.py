import pytest
from autoemulate.core.tuner import Tuner
from autoemulate.core.types import TensorLike
from autoemulate.emulators.gradient_boosting import GradientBoosting


def test_predict_gb(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    gb = GradientBoosting(x, y)
    gb.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = gb.predict(x2)
    assert isinstance(y_pred, TensorLike)
    assert not y_pred.requires_grad

    with pytest.raises(ValueError, match="Gradient calculation is not supported."):
        gb.predict(x2, with_grad=True)


def test_tune_gb(sample_data_y1d):
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5)
    scores, params_list = tuner.run(GradientBoosting)
    assert len(scores) == 5
    assert len(params_list) == 5


# TODO: add determinism test after #512 merged
