import pytest
import torch
import torch.nn as nn

from autoemulate.emulators.neural_networks.cnp_module import Decoder
from autoemulate.emulators.neural_networks.cnp_module import Encoder


# encoder ----------------------------


@pytest.fixture
def encoder():
    input_dim = 3
    output_dim = 2
    hidden_dim = 64
    latent_dim = 32
    return Encoder(input_dim, output_dim, hidden_dim, latent_dim)


def test_encoder_initialization(encoder):
    assert isinstance(encoder, nn.Module)
    assert isinstance(encoder.net, nn.Sequential)
    assert len(encoder.net) == 5  # 3 Linear layers and 2 ReLU activations


def test_encoder_forward_shape(encoder):
    batch_size = 10
    n_points = 5
    input_dim = 3
    output_dim = 2

    x_context = torch.randn(batch_size, n_points, input_dim)
    y_context = torch.randn(batch_size, n_points, output_dim)

    r = encoder(x_context, y_context)

    assert r.shape == (batch_size, 32)  # latent_dim = 32


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
        assert r.shape == (batch_size, 32)  # latent_dim = 32


def test_encoder_different_n_points(encoder):
    input_dim = 3
    output_dim = 2

    batch_size = 10
    n_points_list = [1, 5, 10, 20]

    for n_points in n_points_list:
        x_context = torch.randn(batch_size, n_points, input_dim)
        y_context = torch.randn(batch_size, n_points, output_dim)

        r = encoder(x_context, y_context)
        assert r.shape == (batch_size, 32)  # latent_dim = 32


# decoder ----------------------------


@pytest.fixture
def decoder():
    return Decoder(input_dim=2, latent_dim=64, hidden_dim=128, output_dim=1)


def test_decoder_initialization(decoder):
    assert isinstance(decoder, Decoder)
    assert isinstance(decoder.net, torch.nn.Sequential)
    assert isinstance(decoder.mean_head, torch.nn.Linear)
    assert isinstance(decoder.logvar_head, torch.nn.Linear)


def test_decoder_forward_pass(decoder):
    batch_size, n_points, input_dim = 10, 5, 2
    latent_dim = 64

    r = torch.randn(batch_size, latent_dim)
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
            r = torch.randn(batch_size, latent_dim)
            x_target = torch.randn(batch_size, n_points, input_dim)

            mean, logvar = decoder(r, x_target)

            assert mean.shape == (batch_size, n_points, 1)
            assert logvar.shape == (batch_size, n_points, 1)
