import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Transform, constraints

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import DeviceLike, TensorLike
from autoemulate.data.utils import set_random_seed
from autoemulate.transforms.base import AutoEmulateTransform


class VAE(nn.Module, TorchDeviceMixin):
    """
    Variational Autoencoder implementation in PyTorch.

    Parameters
    ----------
    input_dim: int
        Dimensionality of input data.
    hidden_layers: list
        List of hidden dimensions for encoder and decoder networks.
    latent_dim: int
        Dimensionality of the latent space.
    device: DeviceLike | None
        Device to run the model on. If None, uses the default device (usually CPU or
        GPU). Defaults to None.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list,
        latent_dim: int,
        device: DeviceLike | None = None,
    ):
        nn.Module.__init__(self)
        TorchDeviceMixin.__init__(self, device=device)

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, dim, device=self.device))
            encoder_layers.append(nn.ReLU())
            prev_dim = dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_layers[-1], latent_dim, device=self.device)
        self.fc_var = nn.Linear(hidden_layers[-1], latent_dim, device=self.device)

        # Decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, dim, device=self.device))
            decoder_layers.append(nn.ReLU())
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim, device=self.device))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Encode input to mean and log variance of latent distribution."""
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent representation back to original space."""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass through the VAE."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        """Calculate VAE loss: reconstruction + KL divergence."""
        # MSE reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + beta * kl_loss


class VAETransform(AutoEmulateTransform):
    """VAE transform for dimensionality reduction."""

    domain = constraints.real
    codomain = constraints.real
    bijective = False
    vae: VAE | None = None

    def __init__(  # noqa: PLR0913
        self,
        latent_dim: int = 3,
        hidden_layers: list[int] | None = None,
        epochs: int = 800,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        random_seed: int | None = None,
        beta: float = 1.0,
        verbose: bool = False,
        cache_size: int = 0,
    ):
        """
        Initialize the VAE transform parameters.

        Intialize the VAE transform parameters but defer intialization of the inner
        VAE model until fit is called when the input data is available.

        Parameters
        ----------
        latent_dim: int
            The dimensionality of the VAE latent space. Defaults to 3.
        hidden_layers: list of int
            The number of hidden layers and their sizes in the VAE. If None, defaults to
            [64, 32]. Defaults to None.
        epochs: int
            The number of training epochs for the VAE. Defaults to 800.
        batch_size: int
            The batch size for training the VAE. Defaults to 32.
        learning_rate: float
            The learning rate for the VAE optimizer. Defaults to 1e-3.
        random_seed: int
            Random seed for reproducibility. Defaults to None.
        beta: float
            The beta parameter for the VAE loss function, controlling the trade-off
            between reconstruction loss and KL divergence. Defaults to 1.0.
        verbose: bool
            If True, log training progress. Defaults to False.
        cache_size: int
            Whether to cache previous transform. Set to 0 to disable caching. Set to
            1 to enable caching of the last single value. This might be useful for
            repeated expensive calls with the same input data but is by default
            disabled. See `PyTorch documentation <https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributions/transforms.py#L46-L89>`_
            for more details on caching. Defaults to 0.
        """
        Transform.__init__(self, cache_size=cache_size)
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers or [64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.random_seed = random_seed
        self.verbose = verbose
        self.cache_size = cache_size  # Store for serialization

        # Initialized during fit
        self.vae = None
        self.input_dim = None
        self._is_fitted = False

    def _init_vae(self, intput_dim: int):
        self.input_dim = intput_dim
        self.vae = VAE(
            intput_dim, self.hidden_layers, self.latent_dim, device=self.device
        ).to(self.device)

    def fit(self, x: TensorLike):
        """Fit the VAE on the training data."""
        TorchDeviceMixin.__init__(self, device=x.device)

        # Set random seed for reproducibility
        if self.random_seed is not None:
            set_random_seed(self.random_seed)

        # Create dataloader
        data_loader = self._convert_to_dataloader(
            x, x, batch_size=self.batch_size, shuffle=True
        )

        # Initialize the model
        self._init_vae(intput_dim=x.shape[1])
        assert self.vae is not None

        # Train the VAE
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        self.vae.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in data_loader:
                recon_batch, mu, log_var = self.vae(batch_x)
                loss = self.vae.loss_function(
                    recon_batch, batch_x, mu, log_var, beta=self.beta
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Log progress
            if self.verbose and (epoch + 1) % 10 == 0:
                msg = (
                    f"Epoch {epoch + 1}/{self.epochs}, "
                    f"Loss: {total_loss / len(data_loader):.4f}"
                )
                logging.info(msg)

        self._is_fitted = True

    def _call(self, x):
        self._check_is_fitted()
        assert self.vae is not None
        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(x)
            return mu

    def _inverse(self, y):
        self._check_is_fitted()
        assert self.vae is not None
        self.vae.eval()
        with torch.no_grad():
            return self.vae.decode(y)

    def log_abs_det_jacobian(self, x, y):
        """Log abs det Jacobian not computable since transform is not bijective."""
        _, _ = x, y
        msg = "log det Jacobian not computable since transform is not bijective."
        raise RuntimeError(msg)

    def _jacobian(self, y: TensorLike) -> TensorLike:
        # Delta method to computing covariance in original space of transform's domain
        # requires the calculation of the jacobian of the transform.
        # https://github.com/alan-turing-institute/autoemulate/issues/376#issuecomment-2891374970
        self._check_is_fitted()
        assert self.vae is not None
        # Ensure the input tensor requires gradient for jacobian computation
        if not y.requires_grad:
            y = y.detach().clone().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(self.vae.decode, y)
        assert isinstance(jacobian, TensorLike)
        return jacobian

    def _batch_basis_matrix(self, y):
        n = y.shape[0]
        jacobian = self._jacobian(y)

        # Reshape jacobian to a batch local basis matrices by stacking along the diag
        return torch.stack([jacobian[i, :, i, :] for i in range(n)], 0)

    def _expanded_basis_matrix(self, y):
        jacobian = self._jacobian(y)
        n = y.shape[0]

        # Reshape jacobian for shape of cov_y (n_tasks x n_samples)
        return jacobian.reshape(n * jacobian.shape[1], -1)
