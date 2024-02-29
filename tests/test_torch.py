import numpy as np
import pytest

from autoemulate.compare import AutoEmulate
from autoemulate.emulators import NeuralNetTorch
from autoemulate.experimental_design import LatinHypercube
from autoemulate.utils import get_model_name

# def simple_sim(params):
#     """A simple simulator."""
#     x, y = params
#     return x + 2 * y


# # fixture for simulation input and output
# @pytest.fixture(scope="module")
# def simulation_io():
#     """Setup for tests (Arrange)"""
#     lh = LatinHypercube([(0.0, 1.0), (10.0, 100.0)])
#     sim_in = lh.sample(10)
#     sim_out = [simple_sim(p) for p in sim_in]
#     return sim_in, sim_out


# @pytest.fixture(scope="module")
# def nn_torch_model(simulation_io):
#     """Setup for tests (Arrange)"""
#     sim_in, sim_out = simulation_io
#     nn_torch = NeuralNetTorch(module="mlp")
#     sim_in = sim_in.astype(np.float32)
#     sim_out = np.array(sim_out, dtype=np.float32)
#     nn_torch.fit(sim_in, sim_out)
#     return nn_torch


def test_nn_torch_initialisation():
    nn_torch = NeuralNetTorch()
    assert nn_torch is not None
    assert not hasattr(nn_torch, "module_")


def test_nn_torch_module_initialisation():
    for module in ("mlp", "rbf"):
        nn_torch = NeuralNetTorch(module=module)
        assert nn_torch is not None
        assert not hasattr(nn_torch, "module_")
        del nn_torch


def test_nn_torch_pred_exists():
    input_size, output_size = 10, 2
    X = np.random.rand(100, input_size)
    y = np.random.rand(100, output_size)
    nn_torch_model = NeuralNetTorch(module="mlp")
    nn_torch_model.fit(X, y)
    predictions = nn_torch_model.predict(X)
    assert predictions is not None


def test_nn_torch_pred_len():
    input_size, output_size = 10, 2
    X = np.random.rand(100, input_size)
    y = np.random.rand(100, output_size)
    nn_torch_model = NeuralNetTorch(module="mlp")
    nn_torch_model.fit(X, y)
    predictions = nn_torch_model.predict(X)
    assert len(predictions) == len(y)


def test_nn_torch_pred_type():
    input_size, output_size = 10, 2
    X = np.random.rand(100, input_size)
    y = np.random.rand(100, output_size)
    nn_torch_model = NeuralNetTorch(module="mlp")
    nn_torch_model.fit(X, y)
    predictions = nn_torch_model.predict(X)
    assert isinstance(predictions, np.ndarray)


def test_nn_torch_shape_setter():
    input_size, output_size = 10, 2
    X = np.random.rand(100, input_size)
    y = np.random.rand(100, output_size)
    nn_torch_model = NeuralNetTorch(module="mlp")
    assert nn_torch_model.module__input_size is None
    assert nn_torch_model.module__output_size is None
    nn_torch_model.fit(X, y)
    assert nn_torch_model.module__input_size == input_size
    assert nn_torch_model.n_features_in_ == input_size
    assert nn_torch_model.module_.model[0].in_features == input_size
    assert nn_torch_model.module__output_size == output_size
    assert nn_torch_model.module_.model[-1].out_features == output_size


def test_nn_torch_mismatch_shape():
    input_size, output_size = 10, 2
    X = np.random.rand(100, input_size)
    y = np.random.rand(100, output_size)
    nn_torch_model = NeuralNetTorch(
        module="mlp", module__input_size=5, module__output_size=1
    )
    assert nn_torch_model.module__input_size == 5
    assert nn_torch_model.module__output_size == 1
    assert not hasattr(nn_torch_model, "n_features_in_")
    nn_torch_model.fit(X, y)
    assert nn_torch_model.module__input_size == input_size
    assert nn_torch_model.n_features_in_ == input_size
    assert nn_torch_model.module_.model[0].in_features == input_size
    assert nn_torch_model.module__output_size == output_size
    assert nn_torch_model.module_.model[-1].out_features == output_size


def test_nn_torch_module_methods():
    input_size, output_size = 10, 2
    X = np.random.rand(100, input_size)
    y = np.random.rand(100, output_size)
    nn_torch_model = NeuralNetTorch(
        module="mlp",
        module__input_size=input_size,
        module__output_size=output_size,
    )
    nn_torch_model.fit(X, y)
    assert callable(getattr(nn_torch_model, "get_grid_params"))
    assert callable(getattr(nn_torch_model.module_, "forward"))
    assert callable(getattr(nn_torch_model.module_, "get_grid_params"))


def test_nn_torch_module_grid_params():
    # ensure get_grid_params returns search space even if module is not initialized
    nn_torch_model = NeuralNetTorch(module="mlp")
    assert not hasattr(nn_torch_model, "module_")
    assert callable(getattr(nn_torch_model, "get_grid_params"))
    assert callable(getattr(nn_torch_model.module, "get_grid_params"))


def test_nn_torch_module_ui():
    input_size, output_size = 10, 2
    X = np.random.rand(100, input_size)
    y = np.random.rand(100, output_size)
    em = AutoEmulate()
    em.setup(X, y, model_subset=["NeuralNet"])
    # check that compare does not raise an error
    best = em.compare()


def test_nn_torch_module_ui_param_search():
    input_size, output_size = 10, 2
    X = np.random.rand(100, input_size)
    y = np.random.rand(100, output_size)
    em = AutoEmulate()
    em.setup(X, y, model_subset=["NeuralNet"], param_search=True, param_search_iters=2)
    # check that compare does not raise an error
    best = em.compare()
