import pytest

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoemulate.utils import get_model_name, get_model_param_grid, adjust_param_grid
from autoemulate.emulators import GradientBoosting


# test retrieving model name ---------------------------------------------------
@pytest.fixture()
def model_name():
    return "GradientBoostingRegressor"


def test_basic(model_name):
    gb = GradientBoostingRegressor()
    assert get_model_name(gb) == model_name


def test_multioutput(model_name):
    gb = MultiOutputRegressor(GradientBoostingRegressor())
    assert get_model_name(gb) == model_name


def test_pipeline(model_name):
    gb = Pipeline([("model", GradientBoostingRegressor())])
    assert get_model_name(gb) == model_name


def test_pipeline_multioutput():
    gb = Pipeline([("model", MultiOutputRegressor(GradientBoostingRegressor()))])
    assert get_model_name(gb) == "GradientBoostingRegressor"


def test_pipeline_with_scaler(model_name):
    gb = Pipeline(
        [("scaler", StandardScaler()), ("model", GradientBoostingRegressor())]
    )
    assert get_model_name(gb) == model_name


# test retrieving and adjusting parameter grids ---------------------------------


@pytest.fixture
def model():
    return GradientBoosting()


# gets all the parameters of the model with the sklearn get_params() method
@pytest.fixture
def param_grid(model):
    return model.get_params()


@pytest.fixture
def model_in_pipeline(model):
    return Pipeline([("model", model)])


@pytest.fixture
def model_in_pipe_w_scaler(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


@pytest.fixture
def model_multiout(model):
    return MultiOutputRegressor(model)


@pytest.fixture
def model_multiout_pipe(model):
    return Pipeline([("model", MultiOutputRegressor(model))])


@pytest.fixture
def model():
    return GradientBoosting()


# gets all the parameters of the model with the sklearn get_params() method
@pytest.fixture
def param_grid(model):
    return model.get_params()


@pytest.fixture
def model_in_pipe(model):
    return Pipeline([("model", model)])


@pytest.fixture
def model_in_pipe_w_scaler(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


@pytest.fixture
def model_multiout(model):
    return MultiOutputRegressor(model)


@pytest.fixture
def model_multiout_pipe(model):
    return Pipeline([("model", MultiOutputRegressor(model))])


# test get_model_param_grid ----------------------------------------------------
def test_param_basic(model, param_grid):
    model_grid = get_model_param_grid(model)
    # check that all keys in model_grid are in param_grid
    assert all(key in param_grid.keys() for key in model_grid.keys())


def test_param_pipe(model_in_pipe, param_grid):
    model_grid = get_model_param_grid(model_in_pipe)
    assert all(key in param_grid.keys() for key in model_grid.keys())


def test_param_pipe_scaler(model_in_pipe_w_scaler, param_grid):
    model_grid = get_model_param_grid(model_in_pipe_w_scaler)
    assert all(key in param_grid.keys() for key in model_grid.keys())


def test_param_multiout(model_multiout, param_grid):
    model_grid = get_model_param_grid(model_multiout)
    assert all(key in param_grid.keys() for key in model_grid.keys())


def test_param_multiout_pipe(model_multiout_pipe, param_grid):
    model_grid = get_model_param_grid(model_multiout_pipe)
    assert all(key in param_grid.keys() for key in model_grid.keys())


# test adjust_param_grid -------------------------------------------------------
def test_adj_param_basic(model, param_grid):
    adjusted_param_grid = adjust_param_grid(model, param_grid)
    assert all(key in param_grid.keys() for key in adjusted_param_grid.keys())


def test_adj_param_pipe(model_in_pipe, param_grid):
    adjusted_param_grid = adjust_param_grid(model_in_pipe, param_grid)
    assert all(key.startswith("model__") for key in adjusted_param_grid.keys())


def test_adj_param_pipe_scaler(model_in_pipe_w_scaler, param_grid):
    adjusted_param_grid = adjust_param_grid(model_in_pipe_w_scaler, param_grid)
    assert all(key.startswith("model__") for key in adjusted_param_grid.keys())


def test_adj_param_multiout(model_multiout, param_grid):
    adjusted_param_grid = adjust_param_grid(model_multiout, param_grid)
    assert all(key.startswith("estimator__") for key in adjusted_param_grid.keys())


def test_adj_param_multiout_pipe(model_multiout_pipe, param_grid):
    adjusted_param_grid = adjust_param_grid(model_multiout_pipe, param_grid)
    assert all(
        key.startswith("model__estimator__") for key in adjusted_param_grid.keys()
    )
