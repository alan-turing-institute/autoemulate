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
    y = np.random.rand(30, 1)  # Make y 2D
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
    assert hasattr(em, "preprocessing_results")  # Changed from cv_results
    assert "None" in em.preprocessing_results  # Check default preprocessing

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
        scale_output=False  # Disable output scaling for test
    )
    em.compare()
    
    # Check that at least one model completed successfully
    assert any(len(prep_data["cv_results"]) > 0 
              for prep_data in em.preprocessing_results.values())

def test_run_parallel(Xy_single, kfold):
    X, y = Xy_single
    em = AutoEmulate()
    em.setup(X, y, print_setup=False, cross_validator=kfold, n_jobs=2)
    em.compare()
    assert em.best_model is not None
    # Basic check that parallel worked - models should be fitted
    assert all(len(prep["models"]) > 0 
              for prep in em.preprocessing_results.values())