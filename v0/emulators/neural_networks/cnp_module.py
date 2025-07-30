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
    ):
        super().__init__()
        layers = [nn.Linear(input_dim + output_dim, hidden_dim), activation()]
        for _ in range(hidden_layers_enc):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_context, y_context, context_mask=None):
        """
        Encode context

        Parameters
        ----------
        x_context: (batch_size, n_context_points, input_dim)
        y_context: (batch_size, n_context_points, output_dim)
        context_mask: (batch_size, n_context_points)

        Returns
        -------
        r: (batch_size, 1, latent_dim)
        """
        x = torch.cat([x_context, y_context], dim=-1)
        x = self.net(x)

        if context_mask is not None:
            masked_x = x * context_mask.unsqueeze(-1)
            r = masked_x.sum(dim=1, keepdim=True) / context_mask.sum(
                dim=1, keepdim=True
            ).unsqueeze(
                -1
            )  # mean over valid context points
        else:
            r = x.mean(dim=1, keepdim=True)  # mean over context points
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
        r: (batch_size, 1, latent_dim)
        x_target: (batch_size, n_points, input_dim)

        Returns
        -------
        mean: (batch_size, n_points, output_dim)
        logvar: (batch_size, n_points, output_dim)
        """
        _, n, _ = x_target.shape  # batch_size, n_points, input_dim
        r_expanded = r.expand(-1, n, -1)
        x = torch.cat([r_expanded, x_target], dim=-1)
        hidden = self.net(x)
        mean = self.mean_head(hidden)
        logvar = self.logvar_head(hidden)

        return mean, logvar


class CNPModule(nn.Module):
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
        X_target: (batch_size, n_points, input_dim)
        context_mask: (batch_size, n_context_points), currently unused,
        as we pad with 0's and don't have attention, layernorm yet.

        Returns
        -------
        mean: (batch_size, n_points, output_dim)
        logvar: (batch_size, n_points, output_dim)
        """
        r = self.encoder(X_context, y_context, context_mask)
        mean, logvar = self.decoder(r, X_target)
        return mean, logvar
