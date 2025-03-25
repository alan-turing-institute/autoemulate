import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def get_dim_reducer(
    name,
    n_components=8,
    encoding_dim=8,
    hidden_layers=None,
    epochs=1200,
    batch_size=32,
    beta=1.0,
    verbose=False,
):
    """
    Factory function to get a dimensionality reducer based on name.

    Parameters
    ----------
    name : str or None
        Name of the dimensionality reducer to use.
        Options:
        - 'PCA': Principal Component Analysis
        - 'AE': Autoencoder
        - 'VAE': Variational Autoencoder
        - None: No dimensionality reduction (returns None)

    Returns
    -------
    dim_reducer : object or None
        Scikit-learn compatible dimensionality reducer or None if name is None.
    """
    if name is None:
        return None

    # Return the appropriate dimensionality reducer
    if name == "PCA":
        return TargetPCA(n_components)

    elif name == "VAE":
        return VAEOutputPreprocessor(
            latent_dim=3,
            hidden_dims=[64, 32],
            epochs=800,
            batch_size=32,
            learning_rate=1e-3,
            device=None,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown dimensionality reducer: {name}")


class TargetPCA(BaseEstimator, TransformerMixin):
    """PCA transformer for target values (y) that strictly requires y."""

    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def _validate_y(self, y):
        """Validate that y is provided and properly shaped."""
        if y is None:
            raise ValueError("This is a target transformer - y cannot be None")
        return y.reshape(-1, 1) if len(np.array(y).shape) == 1 else y

    def fit(self, X, y=None):
        y_reshaped = self._validate_y(y)
        self.pca.fit(y_reshaped)
        return self

    def transform(self, X, y=None):
        y_reshaped = self._validate_y(y)
        y_transformed = self.pca.transform(y_reshaped)
        return X, y_transformed

    def fit_transform(self, X, y=None):
        y_reshaped = self._validate_y(y)
        y_transformed = self.pca.fit_transform(y_reshaped)
        return X, y_transformed

    def inverse_transform(self, X, y=None):
        y_reshaped = self._validate_y(y)
        y_original = self.pca.inverse_transform(y_reshaped)
        return X, y_original


class VAEOutputPreprocessor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that applies a Variational Autoencoder (VAE)
    to target values (y) in a sklearn pipeline.

    This class wraps a PyTorch VAE model to preprocess output data through the latent space
    and provides inverse transformation capabilities.

    Parameters
    ----------
    latent_dim : int, default=3
        Dimension of the latent space.
    hidden_dims : list, default=[64, 32]
        Hidden dimensions for the VAE encoder and decoder networks.
    epochs : int, default=100
        Number of training epochs for the VAE.
    batch_size : int, default=32
        Batch size for training.
    learning_rate : float, default=1e-3
        Learning rate for the Adam optimizer.
    device : str, default='cuda' if available else 'cpu'
        Device to use for computation.
    """

    def __init__(
        self,
        latent_dim=3,
        hidden_dims=None,
        epochs=800,
        batch_size=32,
        learning_rate=1e-3,
        device=None,
        verbose=False,
    ):
        self.latent_dim = latent_dim
        self.hidden_dims = [64, 32] if hidden_dims is None else hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.vae = None
        self.is_fitted_ = False

    def _init_vae(self, input_dim):
        """Initialize the VAE model with the appropriate input dimension."""

        self.vae = VAE(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
        ).to(self.device)

    def _create_data_loader(self, y):
        """Create a PyTorch DataLoader for the target values."""
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = torch.utils.data.TensorDataset(y_tensor)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

    def fit(self, X, y=None, **fit_params):
        """
        Fit the VAE model to the target values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data (not used but required by the sklearn API).
        y : array-like, shape (n_samples, n_outputs)
            Target values to transform.

        Returns
        -------
        self : object
            Returns self.
        """
        if y is None:
            raise ValueError(
                "Target values (y) are required for VAEOutputPreprocessor."
            )

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Initialize VAE with input dimension from y
        self.n_features_out_ = y.shape[1]
        self._init_vae(input_dim=self.n_features_out_)

        # Create data loader
        data_loader = self._create_data_loader(y)

        # Train the VAE
        optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        self.vae.train()

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in data_loader:
                y_batch = batch[0]

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                recon_y, mu, log_var = self.vae(y_batch)

                # Calculate loss
                loss = self.vae.loss_function(recon_y, y_batch, mu, log_var)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(data_loader):.4f}"
                )

        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        """
        Transform the target values using the VAE's latent representation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data (passed through unchanged).
        y : array-like, shape (n_samples, n_outputs)
            Target values to transform.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Same as input X (unchanged).
        y_new : array-like, shape (n_samples, latent_dim)
            Transformed target values (encoded to latent space).
        """
        if not self.is_fitted_:
            raise ValueError("The VAEOutputPreprocessor has not been fitted yet.")

        if y is None:
            return X

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Convert to torch tensor
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Set model to evaluation mode
        self.vae.eval()

        # Get latent representation (mu part only, no sampling in transform)
        with torch.no_grad():
            mu, _ = self.vae.encode(y_tensor)
            y_transformed = mu.cpu().numpy()

        return X, y_transformed

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        """
        Inverse transform from latent space back to original target space.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data (passed through unchanged).
        y : array-like, shape (n_samples, latent_dim)
            Latent representations to inverse transform.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Same as input X (unchanged).
        y_new : array-like, shape (n_samples, n_original_outputs)
            Reconstructed target values in original space.
        """
        if not self.is_fitted_:
            raise ValueError("The VAEOutputPreprocessor has not been fitted yet.")

        if y is None:
            return X

        # Convert to torch tensor
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Set model to evaluation mode
        self.vae.eval()

        # Decode from latent representation
        with torch.no_grad():
            y_reconstructed = self.vae.decode(y_tensor).cpu().numpy()

        return X, y_reconstructed

    @property
    def verbose(self):
        """Get verbosity setting."""
        return getattr(self, "_verbose", False)

    @verbose.setter
    def verbose(self, value):
        """Set verbosity setting."""
        self._verbose = value


class VAE(nn.Module):
    """
    Variational Autoencoder implementation in PyTorch.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input data.
    hidden_dims : list
        List of hidden dimensions for encoder and decoder networks.
    latent_dim : int
        Dimensionality of the latent space.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

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
        z = mu + eps * std
        return z

    def decode(self, z):
        """Decode latent representation back to original space."""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass through the VAE."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        """Calculate VAE loss: reconstruction + KL divergence."""
        # MSE reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + kl_loss

def reconstruct_mean_std(y_pred, y_std, transformer, n_samples=1000):
    """
    Sample from a normal distribution for each simulation and reduced dimension.
    
    Parameters:
    -----------
    y_pred : np.ndarray
        Predicted values with shape (n_simulations, reduced_dim)
    y_std : np.ndarray
        Standard deviation values with shape (n_simulations, reduced_dim)
    n_samples : int
        Number of samples to generate for each simulation
    
    Returns:
    --------
    np.ndarray
        Sampled values with shape (n_simulations, n_samples, reduced_dim)
    """

    #if transformer is PCA (so linear), you can use inverse_transform for both mean and std
    #otherwise use sampling procedure to estimate mean and std
    if isinstance(transformer, TargetPCA):
        print("Using PCA")
        pred_mean = transformer.inverse_transform(None, y_pred)[1]
        pred_std = transformer.inverse_transform(None, y_std)[1]

    else:
        samples = np.zeros((y_pred.shape[0], n_samples, y_pred.shape[1]))
        
        for i in range(y_pred.shape[0]):
            # Sample from normal distribution for each simulation
            samples[i] = np.random.normal(
                loc=y_pred[i], 
                scale=y_std[i], 
                size=(n_samples, y_pred.shape[1])
            )
        
        if transformer is not None and hasattr(transformer, "inverse_transform"):
            samples = transformer.inverse_transform(None, samples)[1]

        pred_mean, pred_std = np.mean(samples, axis=1), np.std(samples, axis=1)

    return pred_mean, pred_std