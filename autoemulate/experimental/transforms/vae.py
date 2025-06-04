import logging

import torch
from torch.distributions import Transform, constraints
from torch.utils.data import DataLoader, TensorDataset

from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.types import TensorLike
from autoemulate.preprocess_target import VAE


class VAETransform(AutoEmulateTransform):
    """
    VAE transform for dimensionality reduction.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = False
    vae: VAE | None = None

    def __init__(  # noqa: PLR0913
        self,
        latent_dim=3,
        hidden_layers=None,
        epochs=800,
        batch_size=32,
        learning_rate=1e-3,
        random_state=None,
        beta=1.0,
        verbose=False,
        cache_size: int = 0,
    ):
        Transform.__init__(self, cache_size=cache_size)
        self.latent_dim = latent_dim
        self.hidden_layers = [64, 32] if hidden_layers is None else hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.random_state = random_state
        self.verbose = verbose

        # Initialized during fit
        # TODO: consider this can be init here instead of fit
        self.vae = None
        self.input_dim = None
        self.is_fitted_ = False

    def _init_vae(self, intput_dim: int):
        self.input_dim = intput_dim
        self.vae = VAE(intput_dim, self.hidden_layers, self.latent_dim)

    def fit(self, x: TensorLike):
        """
        Fit the VAE on the training data.
        """
        # Set random seed for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Create dataset and dataloader
        dataset = TensorDataset(x, x)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

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

            # TODO: update with logging
            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                msg = (
                    f"Epoch {epoch + 1}/{self.epochs}, "
                    f"Loss: {total_loss / len(data_loader):.4f}"
                )
                logging.info(msg)

        self.is_fitted_ = True

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
        _, _ = x, y
        msg = "log det Jacobian not computable since transform is not bijective."
        raise RuntimeError(msg)

    def _expanded_basis_matrix(self, x):
        # Delta method to compute covariance in original space
        # https://github.com/alan-turing-institute/autoemulate/issues/376#issuecomment-2891374970
        self._check_is_fitted()
        assert self.vae is not None
        jacobian_z = torch.autograd.functional.jacobian(self.vae.decode, x)
        # Reshape jacobian to match the shape of cov_z (n_tasks x n_samples)
        return jacobian_z.view(jacobian_z.shape[0] * jacobian_z.shape[1], -1)
