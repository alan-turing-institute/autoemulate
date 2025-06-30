import logging

import torch
from torch.distributions import Transform, constraints

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.types import TensorLike
from autoemulate.preprocess_target import VAE


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
        """Intialize the VAE transform parameters but defer intialization of the inner
        VAE model until fit is called when the input data is available.

        Parameters
        ----------

        latent_dim : int, default=3
            The dimensionality of the VAE latent space.
        hidden_layers : list of int, default=None
            The number of hidden layers and their sizes in the VAE. If None, defaults to
            [64, 32].
        epochs : int, default=800
            The number of training epochs for the VAE.
        batch_size : int, default=32
            The batch size for training the VAE.
        learning_rate : float, default=1e-3
            The learning rate for the VAE optimizer.
        random_seed : int, default=None
            Random seed for reproducibility.
        beta : float, default=1.0
            The beta parameter for the VAE loss function, controlling the trade-off
            between reconstruction loss and KL divergence.
        verbose : bool, default=False
            If True, log training progress.
        cache_size : int, default=0
            Whether to cache previous transform. Set to 0 to disable caching. Set to
            1 to enable caching of the last single value. This might be useful for
            repeated expensive calls with the same input data but is by default
            disabled. See `PyTorch documentation <https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributions/transforms.py#L46-L89>`_
            for more details on caching.

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
        """
        Fit the VAE on the training data.
        """
        TorchDeviceMixin.__init__(self, device=x.device)

        # Set random seed for reproducibility
        # TODO: update after #512
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

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
        _, _ = x, y
        msg = "log det Jacobian not computable since transform is not bijective."
        raise RuntimeError(msg)

    def _expanded_basis_matrix(self, y):
        # Delta method to compute covariance in original space of transform's domain.
        # https://github.com/alan-turing-institute/autoemulate/issues/376#issuecomment-2891374970
        self._check_is_fitted()
        assert self.vae is not None
        jacobian_y = torch.autograd.functional.jacobian(self.vae.decode, y)
        # Reshape jacobian to match the shape of cov_y (n_tasks x n_samples)
        return jacobian_y.view(jacobian_y.shape[0] * jacobian_y.shape[1], -1)
