import torch
from autoemulate.experimental.emulators.ensemble import DropoutEnsemble, Ensemble
from autoemulate.experimental.emulators.nn.mlp import MLP
from autoemulate.experimental.types import TensorLike


def test_mlp_ensemble():
    # Training data
    x_train = torch.rand(100, 1) * 10
    x_train = x_train[x_train.flatten().argsort(), :]
    y_train = torch.sin(x_train)

    # Ensemble of MLPs
    emulators = []
    for i in range(4):
        torch.manual_seed(i)
        emulator = MLP(x_train, y_train, layer_dims=[100, 100], lr=1e-2)
        emulator.epochs = 100
        emulators.append(emulator)

    # Ensemble emulator
    emulator = Ensemble(emulators)
    emulator.fit(x_train, y_train)

    # Test
    x_test = torch.linspace(0.0, 15.0, steps=1000).reshape(-1, 1)
    y_test_hat = emulator.predict(x_test)
    assert isinstance(y_test_hat, torch.distributions.MultivariateNormal)
    assert isinstance(y_test_hat.loc, TensorLike)
    assert isinstance(y_test_hat.covariance_matrix, TensorLike)
    assert y_test_hat.loc.shape == (1000, 1)
    assert y_test_hat.covariance_matrix.shape == (1000, 1, 1)


def test_dropout_ensemble():
    # Training data
    x_train = torch.rand(100, 1) * 10
    x_train = x_train[x_train.flatten().argsort(), :]
    y_train = torch.sin(x_train)

    # Dropout Ensemble
    emulator = MLP(x_train, y_train, layer_dims=[100, 100], lr=1e-3, dropout_prob=0.10)
    emulator.epochs = 10
    emulator = DropoutEnsemble(emulator, n_samples=20, jitter=1e-7)
    emulator.fit(x_train, y_train)

    # Test
    x_test = torch.linspace(0.0, 15.0, steps=1000).reshape(-1, 1)
    y_test_hat = emulator.predict(x_test)
    assert isinstance(y_test_hat, torch.distributions.MultivariateNormal)
    assert isinstance(y_test_hat.loc, TensorLike)
    assert isinstance(y_test_hat.covariance_matrix, TensorLike)
    assert y_test_hat.loc.shape == (1000, 1)
    assert y_test_hat.covariance_matrix.shape == (1000, 1, 1)


def test_ensemble_ensemble():
    # Training data
    x_train = torch.rand(100, 1) * 10
    x_train = x_train[x_train.flatten().argsort(), :]
    y_train = torch.sin(x_train)

    # Ensemble of ensembles
    emulators = []
    for i in range(2):
        subemulators = []
        for j in range(4):
            torch.manual_seed(i + j)
            emulator = MLP(x_train, y_train, layer_dims=[100, 100], lr=1e-2)
            emulator.epochs = 100
            subemulators.append(emulator)
        emulator = Ensemble(subemulators)
        emulators.append(emulator)

    # Ensemble emulator
    emulator = Ensemble(emulators)
    emulator.fit(x_train, y_train)

    # Test
    x_test = torch.linspace(0.0, 15.0, steps=1000).reshape(-1, 1)
    y_test_hat = emulator.predict(x_test)
    assert isinstance(y_test_hat, torch.distributions.MultivariateNormal)
    assert isinstance(y_test_hat.loc, TensorLike)
    assert isinstance(y_test_hat.covariance_matrix, TensorLike)
    assert y_test_hat.loc.shape == (1000, 1)
    assert y_test_hat.covariance_matrix.shape == (1000, 1, 1)
