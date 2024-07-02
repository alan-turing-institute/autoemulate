import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import loguniform
from skopt.space import Categorical
from skopt.space import Real


class RobustGaussianNLLLoss(nn.Module):
    def forward(self, y_pred, y_true):
        mean, logvar = y_pred
        variance = torch.exp(logvar.clamp(min=-20, max=20)) + 1e-6
        return 0.5 * torch.mean(
            logvar
            + torch.clamp((y_true - mean) ** 2 / variance, max=1e6)
            + torch.log(torch.tensor(2 * np.pi))
        )


# def sum_log_prob(prob, sample):
#     """Compute log probability then sum all but the z_samples and batch."""
#     log_p = prob.log_prob(sample)
#     sum_log_p = log_p.sum(dim=tuple(range(2, log_p.dim())))
#     return sum_log_p

# class CNPLoss(nn.Module):
#     def __init__(self, reduction='mean'):
#         super().__init__()
#         self.reduction = reduction

#     def forward(self, y_pred, y_true):
#         mean, logvar = y_pred

#         # Create a Normal distribution with the predicted mean and variance
#         p_yCc = torch.distributions.Normal(mean, torch.exp(0.5 * logvar))

#         # Calculate the negative log-likelihood
#         nll = -sum_log_prob(p_yCc, y_true)

#         if self.reduction == 'mean':
#             return nll.mean()
#         elif self.reduction == 'sum':
#             return nll.sum()
#         else:
#             return nll


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
        self, input_dim, output_dim, hidden_dim, latent_dim=64, context_proportion=0.5
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, output_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dim, output_dim)
        self.context_proportion = context_proportion

    def forward(self, X, y, X_target=None):
        # X is expected to be a dict containing 'X' and 'y'
        X_data, y_data = X, y
        batch_size = X_data.shape[0]
        num_context = max(1, int(batch_size * self.context_proportion))
        # Randomly select context points
        # context_idx = torch.randperm(batch_size)[:context_points]
        # print(f"context idx shape: {context_idx.shape}")
        # print(f"context_idx: {context_idx}")
        context_x = X_data[:num_context]
        context_y = y_data[:num_context]
        # Encode context points
        r = self.encoder(context_x, context_y).mean(dim=0, keepdim=True)

        # Decode for all target points
        mean, logvar = self.decoder(r, X_data)
        return mean, logvar
