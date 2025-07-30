import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import loguniform


class Encoder(nn.Module):
    """
    Deterministic encoder for conditional neural process model.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        latent_dim,
        hidden_layers_enc,
        activation,
        context_mask=None,
    ):
        super().__init__()
        layers = [nn.Linear(input_dim + output_dim, hidden_dim), activation()]
        for _ in range(hidden_layers_enc):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

        self.x_encoder = nn.Linear(input_dim, latent_dim)

        self.crossattn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=4, batch_first=True
        )

    def forward(self, x_context, y_context, x_target, context_mask=None):
        """
        Encode context

        Parameters
        ----------
        x_context: (batch_size, n_context_points, input_dim)
        y_context: (batch_size, n_context_points, output_dim)
        context_mask: (batch_size, n_context_points)

        Returns
        -------
        r: (batch_size, n_points, latent_dim)
        """
        # context self attention
        x = torch.cat([x_context, y_context], dim=-1)
        r = self.net(x)
        # q, k, v
        x_target_enc = self.x_encoder(x_target)
        x_context_enc = self.x_encoder(x_context)
        if context_mask is not None:
            r, _ = self.crossattn(
                x_target_enc,
                x_context_enc,
                r,
                need_weights=False,
                key_padding_mask=context_mask,
            )
        else:
            r, _ = self.crossattn(x_target_enc, x_context_enc, r, need_weights=False)
        return r


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim,
        output_dim,
        hidden_layers_dec,
        activation,
    ):
        super().__init__()
        layers = [nn.Linear(latent_dim + input_dim, hidden_dim), activation()]
        for _ in range(hidden_layers_dec):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, r, x_target):
        """
        Decode using representation r and target points x_target

        Parameters
        ----------
        r: (batch_size, n_points, latent_dim)
        x_target: (batch_size, n_points, input_dim)

        Returns
        -------
        mean: (batch_size, n_points, output_dim)
        logvar: (batch_size, n_points, output_dim)
        """
        x = torch.cat([r, x_target], dim=-1)
        hidden = self.net(x)
        mean = self.mean_head(hidden)
        logvar = self.logvar_head(hidden)

        return mean, logvar


class AttnCNPModule(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        latent_dim,
        hidden_layers_enc,
        hidden_layers_dec,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.encoder = Encoder(
            input_dim, output_dim, hidden_dim, latent_dim, hidden_layers_enc, activation
        )
        self.decoder = Decoder(
            input_dim, latent_dim, hidden_dim, output_dim, hidden_layers_dec, activation
        )

    def forward(self, X_context, y_context, X_target=None, context_mask=None):
        """

        Parameters
        ----------
        X_context: (batch_size, n_context_points, input_dim)
        y_context: (batch_size, n_context_points, output_dim)
        X_target: (batch_size, n_target_points, input_dim)
        context_mask: (batch_size, n_context_points), currently unused,
        as we pad with 0's and don't have attention, layernorm yet.

        Returns
        -------
        mean: (batch_size, n_points, output_dim)
        logvar: (batch_size, n_points, output_dim)
        """
        # inverse context_mask
        if context_mask is not None:
            context_mask = ~context_mask
        r = self.encoder(X_context, y_context, X_target)
        mean, logvar = self.decoder(r, X_target)
        return mean, logvar
