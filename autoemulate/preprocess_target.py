import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def get_dim_reducer(
    name,
    reduced_dim=8,
    hidden_layers=[64, 32],
    epochs=2000,
    batch_size=32,
    learning_rate=1e-3,
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
    # Return the appropriate dimensionality reducer
    if name == "PCA":
        return PCA(reduced_dim)
        # return PCA(n_components=n_components) #check! Marjan was sayin base class should not have fit and etc.

    elif name == "VAE":
        return TargetVAE(
            latent_dim=reduced_dim,
            hidden_layers=hidden_layers,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=None,
            beta=beta,
            verbose=verbose,
        )
    elif name == "None":
        return NoChangeTransformer()
    else:
        raise ValueError(f"Unknown dimensionality reducer: {name}")


class TargetPCA(BaseEstimator, TransformerMixin):
    """PCA transformer for target values (y) that strictly requires y."""

    def __init__(self, n_components=None):
        self.n_components = n_components

    def _validate_data(self, X, y=None):
        """Validate that X is provided and properly shaped."""
        if X is None:
            raise ValueError("This is a PCA transformer - input data cannot be None")
        return X.reshape(-1, 1) if len(np.array(X).shape) == 1 else X

    def fit(self, X, y=None):
        X_reshaped = self._validate_data(X)
        if self.n_components is None:
            self.n_components = min(X_reshaped.shape)
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(X_reshaped)
        return self

    def transform(self, X, y=None):
        X_reshaped = self._validate_data(X)
        return self._pca.transform(X_reshaped)

    def fit_transform(self, X, y=None, **fit_params):
        X_reshaped = self._validate_data(X)
        if self.n_components is None:
            self.n_components = min(X_reshaped.shape)
        self._pca = PCA(n_components=self.n_components)
        return self._pca.fit_transform(X_reshaped)

    def inverse_transform(self, X, y=None):
        X_reshaped = self._validate_data(X)
        return self._pca.inverse_transform(X_reshaped)

    def inverse_transform_std(self, x_std):
        """Transform standard deviations from latent space to original space."""
        n_pca_components = x_std.shape[1]
        components = self._pca.components_[:n_pca_components]
        x_std_transformed = np.sqrt(
            np.sum((x_std[:, np.newaxis, :] * components.T) ** 2, axis=2)
        )
        return x_std_transformed

    @property
    def pca(self):
        """Public accessor for the PCA model."""
        return self._pca


class TargetVAE(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper for PyTorch Variational Autoencoder to use as dimensionality reducer.
    Implements fit, transform, and inverse_transform methods required for sklearn pipelines.

    Parameters
    ----------
    latent_dim : int, default=3
        Dimension of the encoded representation (latent space)
    hidden_layers : list of int, default=None
        List of hidden layer sizes for encoder
    epochs : int, default=100
        Number of training epochs
    batch_size : int, default=32
        Batch size for training
    learning_rate : float, default=0.001
        Learning rate for optimizer
    beta : float, default=1.0
        Weight for KL divergence in the loss function (beta-VAE)
    device : str, default=None
        Device to run the model on ('cpu' or 'cuda'). If None, use cuda if available.
    random_state : int, default=None
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print training progress
    """

    def __init__(
        self,
        latent_dim=3,
        hidden_layers=None,
        epochs=800,
        batch_size=32,
        learning_rate=1e-3,
        device=None,
        random_state=None,
        beta=1.0,
        verbose=False,
    ):
        self.latent_dim = latent_dim
        self.hidden_layers = [64, 32] if hidden_layers is None else hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.random_state = random_state
        self.verbose = verbose

        # Will be initialized during fit
        self.vae = None
        self.input_dim = None
        self.is_fitted_ = False

    def _init_vae(self, input_dim):
        """Initialize the VAE model with the appropriate input dimension."""

        self.vae = VAE(
            input_dim=input_dim,
            hidden_layers=self.hidden_layers,
            latent_dim=self.latent_dim,
        ).to(self.device)

    def _check_is_fitted(self):
        """Check if the model is fitted."""
        if not self.is_fitted_:
            raise ValueError(
                "The VAEOutputPreprocessor has not been fitted yet."
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    def fit(self, X, y=None, **fit_params):
        """
        Fit the VAE on the training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored. Present for API consistency.

        Returns
        -------
        self : object
            Returns self.
        """
        # Set random seed for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Get input dimension from data
        self.input_dim = X.shape[1]

        # Convert data to numpy array if needed
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        # Create dataset and dataloader
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)  # Input and target are the same
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the model
        self.vae = VAE(
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            latent_dim=self.latent_dim,
        ).to(self.device)

        # Train the VAE
        optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)

        # Training loop
        self.vae.train()

        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in data_loader:
                # Forward pass
                recon_batch, mu, log_var = self.vae(batch_x)

                # Calculate loss
                loss = self.vae.loss_function(
                    recon_batch, batch_x, mu, log_var, beta=self.beta
                )

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(data_loader):.4f}"
                )

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Reduce dimensionality by encoding the data to the latent space.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, encoding_dim)
            Transformed array (means of the latent distributions).
        """
        self._check_is_fitted()

        # Convert data to numpy array if needed
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Encode data
        # Get latent representation (mu part only, no sampling in transform)
        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(X_tensor)
            X_encoded = mu.cpu().numpy()

        return X_encoded

    def fit_transform(self, X, y=None, **fit_params):
        # Fit the model
        self.fit(X, **fit_params)
        # Return the transformed data
        return self.transform(X)

    def inverse_transform(self, X):
        """
        Transform encoded data back to the original space.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, encoding_dim)
            The encoded data (in latent space).

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Data in original space.
        """
        self._check_is_fitted()

        # Convert data to numpy array if needed
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Decode from latent representation
        self.vae.eval()
        with torch.no_grad():
            X_reconstructed = self.vae.decode(X_tensor).cpu().numpy()

        return X_reconstructed

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
    hidden_layers : list
        List of hidden dimensions for encoder and decoder networks.
    latent_dim : int
        Dimensionality of the latent space.
    """

    def __init__(self, input_dim, hidden_layers, latent_dim):
        super(VAE, self).__init__()

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_layers[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_layers[-1], latent_dim)

        # Decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_layers):
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

    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        """Calculate VAE loss: reconstruction + KL divergence."""
        # MSE reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + beta * kl_loss


def inverse_transform_with_std(model, x_latent_pred, x_latent_std, n_samples=1000):
    """
    Transforms uncertainty (standard deviation) from latent space to original space
    using a sampling-based method.

    This approach draws samples from the latent Gaussian distribution
    (defined by the predicted mean and standard deviation), reconstructs
    each sample in the original space, and computes the resulting mean and
    standard deviation.

    Future improvements could include:
    - Analytical propagation for linear reductions (e.g., PCA)
    - Delta method for nonlinear reductions (e.g., VAE)

    Parameters:
    -----------
    y_pred : np.ndarray
        Predicted values with shape (n_simulations, reduced_dim)
    y_std : np.ndarray
        Standard deviation values with shape (n_simulations, reduced_dim)
    n_samples : int
        Number of samples to draw from the latent distribution for each simulation.

    Returns:
    --------
    pred_mean: np.ndarray
        Mean values with shape (n_simulations, original_dim)
    pred_std: np.ndarray
        Standard deviation values with shape (n_simulations, original_dim)
    """

    if len(x_latent_pred.shape) == 1:
        x_latent_pred = x_latent_pred.reshape(-1, 1)
        x_latent_std = x_latent_std.reshape(-1, 1)

    n_simulations, n_features = x_latent_pred.shape
    samples = []

    for i in range(n_simulations):
        # Sample from normal distribution for each simulation
        samples_latent = np.random.normal(
            loc=x_latent_pred[i],
            scale=x_latent_std[i],
            size=(n_samples, n_features),
        )
        samples.append(model.transformer_.inverse_transform(samples_latent))
    samples = np.array(samples)

    x_reconstructed_mean, x_reconstructed_std = np.mean(samples, axis=1), np.std(
        samples, axis=1
    )

    return x_reconstructed_mean, x_reconstructed_std


class NonTrainableTransformer:
    def __init__(self, NT_transformer):
        self.NT_transformer = (
            NT_transformer  # Expect an instance of either targetPCA or targetVAE
        )

    def __repr__(self):
        return str(self.NT_transformer)

    @property
    def base_transformer(self):
        """Get the underlying transformer instance."""
        return self.NT_transformer

    @property
    def base_transformer_name(self):
        """Get the underlying transformer instance."""
        return str(self.NT_transformer)

    def fit(self, X, y=None):
        return

    def transform(self, X, y=None):
        return self.NT_transformer.transform(X)

    def inverse_transform(self, X, y=None):
        return self.NT_transformer.inverse_transform(X)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def inverse_transform_std(self, X):
        return self.NT_transformer.inverse_transform_std(X)


class NoChangeTransformer(BaseEstimator, TransformerMixin):
    """Transformer which does not do any reduction"""

    def __init__(self):
        pass

    @property
    def base_transformer_name(self):
        """Get the underlying transformer instance."""
        return "NoChangeTransformer"

    def _validate_data(self, X):
        return X

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return X

    def inverse_transform(self, X):
        return X

    def inverse_transform_std(self, x_std):
        return x_std


class InputOutputPipeline(TransformedTargetRegressor):
    """Custom TransformedTargetRegressor to handle inverse transformation of standard deviation."""

    def __init__(
        self,
        regressor=None,
        transformer=None,
        func=None,
        inverse_func=None,
        check_inverse=True,
        n_samples=1000,  # Added custom parameter
    ):
        super().__init__(
            regressor=regressor,
            transformer=transformer,
            func=func,
            inverse_func=inverse_func,
            check_inverse=check_inverse,
        )
        self.n_samples = n_samples  # Store custom parameter

    @staticmethod
    def _ensure_2d(arr):
        """Ensure that arr is a 2D array with shape (n_samples, n_features)."""
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            raise ValueError("Input array must be 1D or 2D")
        return arr

    def predict(self, X, **predict_params):
        """
        Predict using the base regressor and inverse transform the predictions.
        Handles both single predictions and tuples (mean, std) from the base regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        **predict_params : dict
            Additional parameters passed to the regressor's predict method.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or tuple (y_pred, std)
            Transformed prediction. If base regressor supports std and return_std is True,
            returns (y_pred, y_std). Otherwise returns just y_pred.
        """
        check_is_fitted(self)

        # Check if the regressor supports return_std
        base_model = self.regressor_.named_steps["model"]
        supports_std = "return_std" in inspect.signature(base_model.predict).parameters

        # Only pass return_std if the regressor supports it
        if supports_std and predict_params.get("return_std", False):
            pred_mean, pred_std = self.regressor_.predict(X, **predict_params)
            pred_mean = self._ensure_2d(pred_mean)
            pred_std = self._ensure_2d(pred_std)
            return self._inverse_transform_with_std(pred_mean, pred_std)

        # Otherwise just predict mean
        pred_mean = self.regressor_.predict(X, **predict_params)
        # pred_mean = self.regressor_.predict(X, **{k: v for k, v in predict_params.items() if k != 'return_std'})
        pred_mean = self._ensure_2d(pred_mean)

        if predict_params.get("return_std", False):
            # If return_std was requested but not supported, return None for std
            return self.transformer_.inverse_transform(pred_mean).squeeze(), None
        return self.transformer_.inverse_transform(pred_mean).squeeze()

    def _inverse_transform_with_std(self, pred_mean, pred_std, n_samples=1000):
        """
        Transforms uncertainty (standard deviation) from latent space to original space
        using a sampling-based method.

        This approach draws samples from the latent Gaussian distribution
        (defined by the predicted mean and standard deviation), reconstructs
        each sample in the original space, and computes the resulting mean and
        standard deviation.

        Future improvements could include:
        - Analytical propagation for linear reductions (e.g., PCA)
        - Delta method for nonlinear reductions (e.g., VAE)


        Parameters
        ----------
        pred_mean : array-like of shape (n_samples, n_outputs)
            Predicted means in transformed space.
        pred_std : array-like of shape (n_samples, n_outputs)
            Predicted standard deviations in transformed space.

        Returns
        -------
        tuple of ndarrays
            (transformed_mean, transformed_std) in original space
        """
        pred_mean = self._ensure_2d(pred_mean)
        pred_std = self._ensure_2d(pred_std)

        n_simulations, n_features = pred_mean.shape
        samples = []

        for i in range(n_simulations):
            # Sample from normal distribution for each simulation
            samples_latent = np.random.normal(
                loc=pred_mean[i],
                scale=pred_std[i],
                size=(self.n_samples, n_features),
            )
            samples.append(self.transformer_.inverse_transform(samples_latent))
        samples = np.array(samples)

        transformed_mean = np.mean(samples, axis=1).squeeze()
        transformed_std = np.std(samples, axis=1).squeeze()

        return transformed_mean, transformed_std
