import numpy as np
import pytest

from autoemulate.compare import AutoEmulate
from autoemulate.emulators import NeuralNetTorch
from autoemulate.utils import set_random_seed


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
    em.setup(X, y, model_subset=["NNMlp"])
    # check that compare does not raise an error
    best = em.compare()


def test_nn_torch_module_ui_param_search_random():
    set_random_seed(1234)
    input_size, output_size = 10, 2
    X = np.random.rand(100, input_size)
    y = np.random.rand(100, output_size)
    em = AutoEmulate()
    em.setup(
        X,
        y,
        model_subset=["NNMlp"],
        param_search=True,
        param_search_type="random",
        param_search_iters=5,
    )
    # check that compare does not raise an error
    best = em.compare()


def test_nn_torch_module_ui_param_search_bayes():
    set_random_seed(1234)
    input_size, output_size = 10, 2
    X = np.random.rand(10, input_size)
    y = np.random.rand(10, output_size)
    em = AutoEmulate()
    em.setup(
        X,
        y,
        model_subset=["NNMlp"],
        param_search=True,
        param_search_type="bayes",
        param_search_iters=5,
    )
    # check that compare does not raise an error
    best = em.compare()
