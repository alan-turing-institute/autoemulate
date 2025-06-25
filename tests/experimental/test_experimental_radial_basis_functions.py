from autoemulate.experimental.emulators.radial_basis_functions import (
    RadialBasisFunctions,
)
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import TensorLike


def test_predict_rbf(sample_data_rbf, new_data_rbf):
    x, y = sample_data_rbf
    rbf = RadialBasisFunctions(x, y)
    rbf.fit(x, y)
    x2, _ = new_data_rbf
    y_pred = rbf.predict(x2)
    assert isinstance(y_pred, TensorLike)
    assert y_pred.shape == (56, 2)


def test_tune_rbf(sample_data_rbf):
    x, y = sample_data_rbf
    tuner = Tuner(x, y, n_iter=5)
    scores, configs = tuner.run(RadialBasisFunctions)
    assert len(scores) == 5
    assert len(configs) == 5


# TODO: add determinism test after #512
