import pytest
import torch
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.model_selection import RandomizedSearchCV

from autoemulate.emulators.conditional_neural_process_attn import (
    AttentiveConditionalNeuralProcess,
)
from autoemulate.emulators.neural_networks.cnp_module_attn import AttnCNPModule
from autoemulate.emulators.neural_networks.cnp_module_attn import Decoder
from autoemulate.emulators.neural_networks.cnp_module_attn import Encoder


# encoder ----------------------------
@pytest.fixture
def encoder():
    input_dim = 3
    output_dim = 2
    hidden_dim = 64
    latent_dim = 32
    hidden_layers_enc = 3
    activation = nn.ReLU
    return Encoder(
        input_dim, output_dim, hidden_dim, latent_dim, hidden_layers_enc, activation
    )


def test_encoder_initialization(encoder):
    assert isinstance(encoder, nn.Module)
    assert isinstance(encoder.net, nn.Sequential)


def test_encoder_forward_shape(encoder):
    batch_size = 10
    n_points = 5
    n_target = 3
    input_dim = 3
    output_dim = 2

    x_context = torch.randn(batch_size, n_points, input_dim)
    y_context = torch.randn(batch_size, n_points, output_dim)
    x_target = torch.randn(batch_size, n_target, input_dim)

    r = encoder(x_context, y_context, x_target)

    assert r.shape == (batch_size, n_target, 32)


def test_encoder_forward_deterministic(encoder):
    batch_size = 10
    n_points = 5
    input_dim = 3
    output_dim = 2
    n_target = 3

    x_context = torch.randn(batch_size, n_points, input_dim)
    y_context = torch.randn(batch_size, n_points, output_dim)
    x_target = torch.randn(batch_size, n_target, input_dim)

    r1 = encoder(x_context, y_context, x_target)
    r2 = encoder(x_context, y_context, x_target)

    assert torch.allclose(r1, r2)


def test_encoder_different_batch_sizes(encoder):
    input_dim = 3
    output_dim = 2

    batch_sizes = [1, 5, 10]
    n_points = 5
    n_target = 3

    for batch_size in batch_sizes:
        x_context = torch.randn(batch_size, n_points, input_dim)
        y_context = torch.randn(batch_size, n_points, output_dim)
        x_target = torch.randn(batch_size, n_target, input_dim)

        r = encoder(x_context, y_context, x_target)
        assert r.shape == (batch_size, n_target, 32)


def test_encoder_mask(encoder):
    input_dim = 5
    output_dim = 3
    hidden_dim = 64
    latent_dim = 32
    hidden_layers_enc = 2
    activation = torch.nn.ReLU

    encoder = Encoder(
        input_dim, output_dim, hidden_dim, latent_dim, hidden_layers_enc, activation
    )

    batch_size = 2
    n_context_points = 10
    n_target_points = 8

    x_context = torch.randn(batch_size, n_context_points, input_dim)
    y_context = torch.randn(batch_size, n_context_points, output_dim)
    x_target = torch.randn(batch_size, n_target_points, input_dim)

    context_mask = torch.ones(batch_size, n_context_points, dtype=torch.bool)
    context_mask[:, -3:] = False

    output_with_mask = encoder(x_context, y_context, x_target, context_mask)
    output_without_mask = encoder(x_context, y_context, x_target)

    assert not torch.allclose(
        output_with_mask, output_without_mask
    ), "Mask doesn't seem to affect the output"

    assert output_with_mask.shape == (batch_size, n_target_points, latent_dim)
    assert output_without_mask.shape == (batch_size, n_target_points, latent_dim)


# decoder ----------------------------
@pytest.fixture
def decoder():
    input_dim = 2
    latent_dim = 64
    hidden_dim = 128
    output_dim = 1
    hidden_layers_dec = 5
    activation = nn.ReLU
    return Decoder(
        input_dim, latent_dim, hidden_dim, output_dim, hidden_layers_dec, activation
    )


def test_decoder_initialization(decoder):
    assert isinstance(decoder, nn.Module)
    assert isinstance(decoder.net, nn.Sequential)
    assert isinstance(decoder.mean_head, nn.Linear)
    assert isinstance(decoder.logvar_head, nn.Linear)


def test_decoder_forward_shape(decoder):
    batch_size, n_points, input_dim = 10, 5, 2
    latent_dim = 64

    r = torch.randn(batch_size, n_points, latent_dim)
    x_target = torch.randn(batch_size, n_points, input_dim)

    mean, logvar = decoder(r, x_target)

    assert mean.shape == (batch_size, n_points, 1)
    assert logvar.shape == (batch_size, n_points, 1)


def test_decoder_different_batch_sizes(decoder):
    latent_dim = 64
    input_dim = 2

    for batch_size in [1, 10, 100]:
        for n_points in [1, 5, 20]:
            r = torch.randn(batch_size, n_points, latent_dim)
            x_target = torch.randn(batch_size, n_points, input_dim)

            mean, logvar = decoder(r, x_target)

            assert mean.shape == (batch_size, n_points, 1)
            assert logvar.shape == (batch_size, n_points, 1)


# attn cnp ----------------------------
@pytest.fixture
def attn_cnp_module():
    return AttnCNPModule(
        input_dim=2,
        output_dim=1,
        hidden_dim=32,
        latent_dim=64,
        hidden_layers_enc=2,
        hidden_layers_dec=2,
        activation=nn.ReLU,
    )


@pytest.fixture
def attn_cnp_module_2d():
    return AttnCNPModule(
        input_dim=2,
        output_dim=2,
        hidden_dim=32,
        latent_dim=64,
        hidden_layers_enc=2,
        hidden_layers_dec=2,
        activation=nn.ReLU,
    )


def test_attn_cnp_module_initialization(attn_cnp_module):
    assert isinstance(attn_cnp_module, AttnCNPModule)
    assert isinstance(attn_cnp_module.encoder, Encoder)
    assert isinstance(attn_cnp_module.decoder, Decoder)


def test_attn_cnp_module_forward_shape(attn_cnp_module):
    n_points = 16
    b, n, dx = 32, n_points, 2
    dy = 1
    X_context = torch.randn(b, n_points, dx)
    y_context = torch.randn(b, n_points, dy)
    X_target = torch.randn(b, n, dx)

    mean, logvar = attn_cnp_module(X_context, y_context, X_target)

    assert mean.shape == (b, n, dy)
    assert logvar.shape == (b, n, dy)


def test_attn_cnp_module_forward_shape_2d(attn_cnp_module_2d):
    n_points = 16
    b, n, dx = 32, n_points, 2
    dy = 2
    X_context = torch.randn(b, n_points, dx)
    y_context = torch.randn(b, n_points, dy)
    X_target = torch.randn(b, n, dx)

    mean, logvar = attn_cnp_module_2d(X_context, y_context, X_target)

    assert mean.shape == (b, n, dy)
    assert logvar.shape == (b, n, dy)


# test whether param search works
def test_attn_cnp_param_search():
    X, y = make_regression(n_samples=30, n_features=5, n_targets=2, random_state=0)
    param_grid = {
        "hidden_dim": [16, 32],
        "latent_dim": [16, 32],
    }

    mod = AttentiveConditionalNeuralProcess()

    grid_search = RandomizedSearchCV(
        estimator=mod, param_distributions=param_grid, cv=3, n_iter=3
    )
    grid_search.fit(X, y)

    assert grid_search.best_score_ > 0.3
    assert grid_search.best_params_["hidden_dim"] in [16, 32]
