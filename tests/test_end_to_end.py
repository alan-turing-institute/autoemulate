import numpy as np
import pytest
from sklearn.model_selection import KFold

from autoemulate.compare import AutoEmulate


@pytest.fixture()
def kfold():
    return KFold(n_splits=2)


@pytest.fixture()
def Xy_single():
    X = np.random.rand(30, 2)
    y = np.random.rand(30)
    return X, y


@pytest.fixture()
def Xy_multi():
    X = np.random.rand(30, 2)
    y = np.random.rand(30, 2)
    return X, y


@pytest.mark.parametrize("Xy", ["Xy_single", "Xy_multi"])
def test_run(Xy, request):
    X, y = request.getfixturevalue(Xy)
    em = AutoEmulate()
    em.setup(X, y, print_setup=False)
    em.compare()
    assert em.best_model is not None
    assert em.cv_results is not None


def test_run_param_search(Xy_single, kfold):
    X, y = Xy_single
    em = AutoEmulate()
    em.setup(
        X,
        y,
        print_setup=False,
        param_search=True,
        param_search_iters=1,
        cross_validator=kfold,
    )
    em.compare()
    assert em.best_model is not None
    assert em.cv_results is not None


def test_run_parallel(Xy_single, kfold):
    X, y = Xy_single
    em = AutoEmulate()
    em.setup(X, y, print_setup=False, cross_validator=kfold, n_jobs=2)
    em.compare()
    assert em.best_model is not None
    assert em.cv_results is not None
