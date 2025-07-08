import pytest
from autoemulate.experimental.emulators.gradient_boosting import GradientBoosting
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import TensorLike


def test_predict_gb(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    gb = GradientBoosting(x, y)
    gb.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = gb.predict(x2)
    assert isinstance(y_pred, TensorLike)
    assert not y_pred.requires_grad

    with pytest.raises(ValueError, match="cannot compute gradients"):
        gb.predict(x2, with_grad=True)


def test_tune_gb(sample_data_y1d):
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5)
    scores, configs = tuner.run(GradientBoosting)
    assert len(scores) == 5
    assert len(configs) == 5


# TODO: add determinism test after #512 merged
