import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

from autoemulate.emulators.conditional_neural_process import ConditionalNeuralProcess
from autoemulate.utils import set_random_seed


class AttentiveConditionalNeuralProcess(ConditionalNeuralProcess):
    def __init__(
        self,
        # architecture
        hidden_dim=64,
        latent_dim=64,
        hidden_layers_enc=3,
        hidden_layers_dec=3,
        # data per episode
        min_context_points=3,
        max_context_points=10,
        n_episode=32,
        # training
        max_epochs=100,
        lr=5e-3,
        batch_size=16,
        activation=nn.ReLU,
        optimizer=torch.optim.AdamW,
        normalize_y=True,
        # misc
        device="cpu",
        random_state=None,
        attention=True,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            hidden_layers_enc=hidden_layers_enc,
            hidden_layers_dec=hidden_layers_dec,
            min_context_points=min_context_points,
            max_context_points=max_context_points,
            n_episode=n_episode,
            max_epochs=max_epochs,
            lr=lr,
            batch_size=batch_size,
            activation=activation,
            optimizer=optimizer,
            normalize_y=normalize_y,
            device=device,
            random_state=random_state,
            attention=attention,
        )
