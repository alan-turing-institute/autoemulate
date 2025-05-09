from autoemulate.experimental.emulators.svm import (
    SupportVectorMachines,
)
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import TensorLike


def test_predict_svm(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    y = y.reshape(-1, 1)
    svm = SupportVectorMachines(x, y)
    svm.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = svm.predict(x2)
    print(type(y_pred))
    assert isinstance(y_pred, TensorLike)


def test_tune_svm(sample_data_y1d):
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5)
    scores, configs = tuner.run(SupportVectorMachines)
    assert len(scores) == 5
    assert len(configs) == 5
