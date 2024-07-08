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
        self.net = nn.Sequential(
            nn.Linear(input_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x_context, y_context):
        """
        Encode context

        Parameters
        ----------
        x_context: (batch_size, n_points, input_dim)
        y_context: (batch_size, n_points, output_dim)

        Returns
        -------
        r: (batch_size, latent_dim)
        """
        x = torch.cat([x_context, y_context], dim=-1)
        x = self.net(x)
        r = x.mean(dim=1, keepdim=True)  # mean over context points
        return r


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
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
        # print(f"mean: {mean.shape}")
        # print(f"logvar: {logvar.shape}")
        # print(f"hidden: {hidden.shape}")
        # sigma = 0.1 + 0.9 * torch.nn.functional.softplus(logvar)
        # dist = torch.distributions.Normal(mean, sigma)
        # Debug prints
        # if torch.isnan(mean).any() or torch.isnan(logvar).any():
        #     print("NaN detected in mean or logvar")
        #     print(f"mean: {mean}")
        #     print(f"logvar: {logvar}")

        return mean, logvar


class CNPModule(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, latent_dim=64, n_context_points=16
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, output_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dim, output_dim)
        self.n_context_points = n_context_points

    def forward(self, X_context, y_context, X_target=None):
        """

        Parameters
        ----------
        X_context: (batch_size, n_points, input_dim)
        y_context: (batch_size, n_points, output_dim)
        X_target: (batch_size, n_sample, input_dim)

        X_target uses all points, as this has shown to be more effect for training

        Returns
        -------
        mean: (batch_size, n_sample, output_dim)
        logvar: (batch_size, n_sample, output_dim)
        """
        # Encode context points
        r = self.encoder(X_context, y_context)  # (batch x latent_dim)
        # Decode for all target points
        mean, logvar = self.decoder(r, X_target)
        return mean, logvar
