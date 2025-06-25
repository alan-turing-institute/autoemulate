import pytest
import torch
import torch.nn as nn

from autoemulate.emulators.neural_networks.cnp_module import CNPModule
from autoemulate.emulators.neural_networks.cnp_module import Decoder
from autoemulate.emulators.neural_networks.cnp_module import Encoder

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
    input_dim = 3
    output_dim = 2

    x_context = torch.randn(batch_size, n_points, input_dim)
    y_context = torch.randn(batch_size, n_points, output_dim)

    r = encoder(x_context, y_context)

    assert r.shape == (batch_size, 1, 32)  # b , n, latent_dim


def test_encoder_forward_deterministic(encoder):
    batch_size = 10
    n_points = 5
    input_dim = 3
    output_dim = 2

    x_context = torch.randn(batch_size, n_points, input_dim)
    y_context = torch.randn(batch_size, n_points, output_dim)

    r1 = encoder(x_context, y_context)
    r2 = encoder(x_context, y_context)

    assert torch.allclose(r1, r2)


def test_encoder_different_batch_sizes(encoder):
    input_dim = 3
    output_dim = 2

    batch_sizes = [1, 5, 10]
    n_points = 5

    for batch_size in batch_sizes:
        x_context = torch.randn(batch_size, n_points, input_dim)
        y_context = torch.randn(batch_size, n_points, output_dim)

        r = encoder(x_context, y_context)
        assert r.shape == (batch_size, 1, 32)  # latent_dim = 32


def test_encoder_different_n_points(encoder):
    input_dim = 3
    output_dim = 2

    batch_size = 10
    n_points_list = [1, 5, 10, 20]

    for n_points in n_points_list:
        x_context = torch.randn(batch_size, n_points, input_dim)
        y_context = torch.randn(batch_size, n_points, output_dim)

        r = encoder(x_context, y_context)
        assert r.shape == (batch_size, 1, 32)  # latent_dim = 32


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
    assert isinstance(decoder, Decoder)
    assert isinstance(decoder.net, torch.nn.Sequential)
    assert isinstance(decoder.mean_head, torch.nn.Linear)
    assert isinstance(decoder.logvar_head, torch.nn.Linear)


def test_decoder_forward_pass(decoder):
    batch_size, n_points, input_dim = 10, 5, 2
    latent_dim = 64

    r = torch.randn(batch_size, 1, latent_dim)
    x_target = torch.randn(batch_size, n_points, input_dim)

    mean, logvar = decoder(r, x_target)

    assert mean.shape == (batch_size, n_points, 1)
    assert logvar.shape == (batch_size, n_points, 1)
    assert not torch.isnan(mean).any()
    assert not torch.isnan(logvar).any()


def test_decoder_different_batch_sizes(decoder):
    latent_dim = 64
    input_dim = 2

    for batch_size in [1, 10, 100]:
        for n_points in [1, 5, 20]:
            r = torch.randn(batch_size, 1, latent_dim)
            x_target = torch.randn(batch_size, n_points, input_dim)

            mean, logvar = decoder(r, x_target)

            assert mean.shape == (batch_size, n_points, 1)
            assert logvar.shape == (batch_size, n_points, 1)


# # cnp ----------------------------


@pytest.fixture
def cnp_module():
    return CNPModule(
        input_dim=2,
        output_dim=1,
        hidden_dim=32,
        latent_dim=64,
        hidden_layers_enc=2,
        hidden_layers_dec=2,
        activation=nn.ReLU,
    )


def test_cnp_module_init(cnp_module):
    assert isinstance(cnp_module, CNPModule)
    assert isinstance(cnp_module.encoder, Encoder)
    assert isinstance(cnp_module.decoder, Decoder)


def test_cnp_module_forward_train(cnp_module):
    n_points = 16
    b, n, dx = 32, n_points, 2
    dy = 1
    X_context = torch.randn(b, n_points, dx)
    y_context = torch.randn(b, n_points, dy)
    X_target = torch.randn(b, n, dx)

    mean, logvar = cnp_module(X_context, y_context, X_target)

    assert mean.shape == (b, n, dy)
    assert logvar.shape == (b, n, dy)


# def test_cnp_module_forward_train_2d(cnp_module):
#     b, n, dx = 32, 24, 2
#     dy = 2
#     X = torch.randn(b, n, dx)
#     y = torch.randn(b, n, dy)
#     # re-initialise with 2 output dims
#     cnp_module = CNPModule(
#         input_dim=2, output_dim=2, hidden_dim=32, latent_dim=64, n_context_points=16
#     )
#     mean, logvar = cnp_module(X, y)
#     assert mean.shape == (b, n, dy)
#     assert logvar.shape == (b, n, dy)
