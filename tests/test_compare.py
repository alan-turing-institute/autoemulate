import pytest
import numpy as np
import pandas as pd
from autoemulate.experimental_design import ExperimentalDesign, LatinHypercube
from autoemulate.emulators import GaussianProcess, RandomForest
from autoemulate.compare import AutoEmulate
from autoemulate.metrics import METRIC_REGISTRY
from autoemulate.emulators import MODEL_REGISTRY
from autoemulate.cv import CV_REGISTRY


@pytest.fixture
def random_data():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    return X, y


@pytest.fixture
def ae_instance(random_data):
    X, y = random_data
    ae = AutoEmulate()
    ae.setup(X, y)
    return ae


@pytest.fixture
def fitted_ae_instance(random_data):
    X, y = random_data
    ae = AutoEmulate()
    ae.setup(X, y)
    ae.compare()
    return ae


def test_initialisation():
    ae = AutoEmulate()
    assert ae.is_set_up == False
    assert isinstance(ae.scores_df, pd.DataFrame)


def test_setup(ae_instance):
    ae = ae_instance
    assert ae.is_set_up == True
    assert len(ae.models) == len(MODEL_REGISTRY.keys())
    assert len(ae.metrics) == len(METRIC_REGISTRY.keys())


def test__score_model_with_cv(fitted_ae_instance):
    ae = fitted_ae_instance
    # test that ae.scores_df is pandas dataframe with correct columns
    assert isinstance(ae.scores_df, pd.DataFrame)
    assert set(ae.scores_df.columns) == set(["model", "metric", "fold", "score"])
