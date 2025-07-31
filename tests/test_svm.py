import pytest
import torch
from autoemulate.core.tuner import Tuner
from autoemulate.core.types import TensorLike
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators.svm import SupportVectorMachine


def test_predict_svm(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    y = y.reshape(-1, 1)
    svm = SupportVectorMachine(x, y)
    svm.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = svm.predict(x2)
    assert isinstance(y_pred, TensorLike)
    assert not y_pred.requires_grad

    with pytest.raises(ValueError, match="Gradient calculation is not supported."):
        svm.predict(x2, with_grad=True)


def test_tune_svm(sample_data_y1d):
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5)
    scores, params_list = tuner.run(SupportVectorMachine)
    assert len(scores) == 5
    assert len(params_list) == 5


def test_svm_deterministic_with_seed(sample_data_y1d, new_data_y1d):
    """
    SVMs are deterministic so we do not expect different outputs for different seeds.
    """
    x, y = sample_data_y1d
    y = y.reshape(-1, 1)
    x2, _ = new_data_y1d
    seed = 42
    set_random_seed(seed)
    model1 = SupportVectorMachine(x, y)
    model1.fit(x, y)
    pred1 = model1.predict(x2)
    new_seed = 24
    set_random_seed(new_seed)
    model2 = SupportVectorMachine(x, y)
    model2.fit(x, y)
    pred2 = model2.predict(x2)
    assert isinstance(pred1, TensorLike)
    assert isinstance(pred2, TensorLike)
    assert torch.allclose(pred1, pred2)
