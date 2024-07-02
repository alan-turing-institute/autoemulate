import numpy as np
import torch
import torch.nn as nn
from scipy.stats import loguniform
from skopt.space import Categorical
from skopt.space import Real


class GaussianNLLLoss(nn.Module):
    def forward(self, y_pred, y_true):
        mean, logvar = y_pred
        variance = torch.exp(logvar)
        return 0.5 * torch.mean(
            logvar
            + (y_true - mean) ** 2 / variance
            + torch.log(torch.tensor(2 * np.pi))
        )


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x, y):
        input_pairs = torch.cat([x, y], dim=-1)
        return self.network(input_pairs)


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, r, x_target):
        input = torch.cat([r.expand(x_target.shape[0], -1), x_target], dim=-1)
        hidden = self.network(input)
        mean = self.mean_head(hidden)
        logvar = self.logvar_head(hidden)
        return mean, logvar


class CNPModule(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, latent_dim=64, context_points=5
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, output_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dim, output_dim)
        self.context_points = context_points

    def forward(self, X, y, X_target=None):
        # X is expected to be a dict containing 'X' and 'y'
        X_data, y_data = X, y
        # Randomly select context points
        context_idx = torch.randperm(X_data.shape[0])[: self.context_points]
        # print(f"context_idx: {context_idx}")
        context_x = X_data[context_idx]
        context_y = y_data[context_idx]
        # Encode context points
        r = self.encoder(context_x, context_y).mean(dim=0, keepdim=True)

        # Decode for all target points
        mean, logvar = self.decoder(r, X_data)
        return mean, logvar
