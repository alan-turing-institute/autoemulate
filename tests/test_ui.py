# end-to-end tests
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from autoemulate.compare import AutoEmulate
from autoemulate.emulators import model_registry


@pytest.fixture
def param_search_ae():
    X = np.random.rand(140, 2)
    y = np.random.rand(140, 1)

    # names of all models
    all_models = list(model_registry.get_model_names().keys())

    ae = AutoEmulate()
    ae.setup(
        X,
        y,
        cross_validator=KFold(n_splits=2),
        param_search_type="random",
        param_search=True,
        param_search_iters=1,
        models=all_models,
    )
    ae.compare()
    return ae


# take fast fitting models for testing
model_subset = ["SecondOrderPolynomial", "RadialBasisFunctions"]


def test_scalers():
    X = np.random.rand(100, 5)
    y = np.random.rand(100, 1)

    scalers = [MinMaxScaler(), RobustScaler()]

    for scaler in scalers:
        ae = AutoEmulate()
        ae.setup(X, y, scaler=scaler, models=model_subset)
        ae.compare()

        assert ae.best_model is not None


def test_dimension_reducers():
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 1)

    dim_reducers = [PCA(n_components=5), KernelPCA(n_components=5)]

    for dim_reducer in dim_reducers:
        ae = AutoEmulate()
        ae.setup(X, y, reduce_dim=True, dim_reducer=dim_reducer, models=model_subset)
        ae.compare()

        assert ae.best_model is not None


def test_cross_validators():
    X = np.random.rand(100, 5)
    y = np.random.rand(100, 1)

    ae = AutoEmulate()
    ae.setup(X, y, cross_validator=KFold(n_splits=5), models=model_subset)
    ae.compare()

    assert ae.best_model is not None


def test_param_search(param_search_ae):
    assert param_search_ae.best_model is not None


def test_save_load_with_param_search(param_search_ae):
    with TemporaryDirectory() as temp_dir:
        for name in param_search_ae.model_names:
            save_path = Path(temp_dir) / f"test_model_{name}"
            param_search_ae.save(param_search_ae.get_model(name), save_path)
            loaded_model = param_search_ae.load(save_path)
            assert loaded_model is not None
