import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from autoemulate.compare import AutoEmulate

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

    cross_validators = [KFold(n_splits=5)]

    for cross_validator in cross_validators:
        ae = AutoEmulate()
        ae.setup(X, y, cross_validator=cross_validator, models=model_subset)
        ae.compare()

        assert ae.best_model is not None
