import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin

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
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + kl_loss

class VAEOutputPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer for sklearn Pipeline that uses a PyTorch Variational Autoencoder (VAE)
    to perform dimensionality reduction on y (output) values during model training.
    
    Parameters
    ----------
    latent_dim : int, default=2
        The dimensionality of the latent space.
    
    epochs : int, default=100
        Number of epochs to train the VAE.
    
    batch_size : int, default=32
        Batch size for training the VAE.
    
    learning_rate : float, default=0.001
        Learning rate for the optimizer.
    
    hidden_dims : list, default=None
        List of hidden dimensions for the encoder and decoder networks.
        If None, [64, 32] will be used.
    
    verbose : bool, default=False
        Whether to print training progress.
        
    device : str, default=None
        Device to use for training ('cuda' or 'cpu'). If None, will use
        CUDA if available, otherwise CPU.
    """
    
    def __init__(self, latent_dim=8, epochs=100, batch_size=32, 
                 learning_rate=0.001, hidden_dims=None, verbose=False,
                 device=None):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dims = hidden_dims if hidden_dims is not None else [32, 16]
        self.verbose = verbose
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted_ = False
        self.y_transformed_ = None  # Store last transformed y for inverse_transform
        
    def fit(self, X, y=None):
        """
        Fit the VAE model to the output data y.
        
        Parameters
        ----------
        X : array-like
            Input features (not used by this transformer)
        y : array-like
            Target values to be preprocessed
            
        Returns
        -------
        self : object
            Returns self.
        """
        if y is None:
            return self
            
        # Convert y to numpy array if needed
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        
        # Store input dimensionality
        self.input_dim_ = y.shape[1] if len(y.shape) > 1 else 1
        
        # Reshape for 1D inputs
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Store original y shape
        self.y_shape_ = y.shape
        
        # Convert to PyTorch tensors
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Create dataset and dataloader
        dataset = TensorDataset(y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model_ = VAE(
            input_dim=self.input_dim_,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim
        ).to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                batch_y = batch[0].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                recon_y, mu, log_var = self.model_(batch_y)
                
                # Compute loss
                loss = self.model_.loss_function(recon_y, batch_y, mu, log_var)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X, y=None):
        """
        Transform input data X (pass-through) and output data y (if provided).
        During fitting in a pipeline, y is provided and transformed to the latent space.
        
        Parameters
        ----------
        X : array-like
            Input features (returned unchanged)
        y : array-like, default=None
            Target values to transform
            
        Returns
        -------
        X : array-like
            Original input features
        y_transformed : array-like or None
            Transformed target values in the latent space (if y was provided)
        """
        # During pipeline.fit(), return X unchanged and transformed y
        if y is not None and self.is_fitted_:
            # Convert y to numpy array if needed
            if not isinstance(y, np.ndarray):
                y = np.asarray(y)
            
            # Store original y for later use in inverse_transform
            self.original_y_ = y.copy()
            
            # Reshape for 1D inputs
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            # Convert to PyTorch tensor
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
            
            # Transform to latent space
            self.model_.eval()
            with torch.no_grad():
                mu, _ = self.model_.encode(y_tensor)
                y_transformed = mu.cpu().numpy()
            
            # Store the transformed y
            self.y_transformed_ = y_transformed
            
            return X, y_transformed
        
        # During pipeline.transform() or pipeline.predict(), return X unchanged
        return X
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : array-like
            Input features
        y : array-like, default=None
            Target values
            
        Returns
        -------
        X : array-like
            Original input features
        y_transformed : array-like or None
            Transformed target values (if y was provided)
        """
        return self.fit(X, y).transform(X, y)
    
    def inverse_transform_y(self, y_transformed):
        """
        Inverse transform the latent space representation back to the original space.
        
        Parameters
        ----------
        y_transformed : array-like
            Transformed target values in the latent space
            
        Returns
        -------
        y : array-like
            Reconstructed target values in the original space
        """
        if not self.is_fitted_:
            return y_transformed
            
        # Convert to numpy array if needed
        if not isinstance(y_transformed, np.ndarray):
            y_transformed = np.asarray(y_transformed)
        
        # Convert to PyTorch tensor
        y_tensor = torch.tensor(y_transformed, dtype=torch.float32).to(self.device)
        
        # Decode from latent space
        self.model_.eval()
        with torch.no_grad():
            y_reconstructed = self.model_.decode(y_tensor).cpu().numpy()
        
        # Reshape if original was 1D
        if self.input_dim_ == 1:
            y_reconstructed = y_reconstructed.ravel()
            
        return y_reconstructed
    
    def inverse_transform(self, X):
        """
        Standard sklearn inverse_transform method.
        Takes transformed X (which in this case is actually the transformed y)
        and returns the original space representation.
        
        Parameters
        ----------
        X : array-like
            Transformed data in the latent space
            
        Returns
        -------
        X_original : array-like
            Data reconstructed back to original space
        """
        # For pipeline compatibility, treat X as the transformed y values
        return self.inverse_transform_y(X)


class OutputOnlyPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer for sklearn Pipeline that applies preprocessing
    to only the y (output) values during model training.
    
    Parameters
    ----------
    methods : str or list of str, default=None
        The preprocessing method(s) to apply to y.
        Options include:
        - 'pca': Apply PCA to reduce dimensionality
        - 'standardize': Apply StandardScaler
        - custom methods can be added
    
    n_components : int or float, default=None
        For PCA, the number of components to keep.
        If None and method includes 'pca', keeps all components.
    
    Attributes
    ----------
    transformers_ : dict
        Dictionary of fitted transformer objects with method names as keys.
    inverse_transformers_ : dict
        Dictionary of inverse transformation methods for each transformer.
    """
    
    def __init__(self, methods=None, n_components=None):
        self.methods = methods
        self.n_components = n_components
        self.transformers_ = {}
        self.inverse_transformers_ = {}
        
    def fit(self, X, y=None):
        """
        Fit all the transformers on the output data.
        
        Parameters
        ----------
        X : array-like
            Input features (not used in this transformer)
        y : array-like
            Target values to be preprocessed
            
        Returns
        -------
        self : object
            Returns self.
        """
        if y is None:
            return self
            
        # Ensure y is a numpy array
        y = np.asarray(y)
        
        # Initialize transformers based on methods
        if self.methods is None:
            return self
            
        methods = self.methods if isinstance(self.methods, list) else [self.methods]
        
        for method in methods:
            if method == 'pca':
                from sklearn.decomposition import PCA
                transformer = PCA(n_components=self.n_components)
                self.transformers_[method] = transformer.fit(y)
                self.inverse_transformers_[method] = lambda y_transformed, transformer=transformer: transformer.inverse_transform(y_transformed)
                
            elif method == 'standardize':
                from sklearn.preprocessing import StandardScaler
                transformer = StandardScaler()
                self.transformers_[method] = transformer.fit(y)
                self.inverse_transformers_[method] = lambda y_transformed, transformer=transformer: transformer.inverse_transform(y_transformed)
        
        # Store original y for reference
        self.original_y_ = y.copy()
        self.y_transformed_ = None
        
        return self
    
    def transform(self, X, y=None):
        """
        Transform input data X (pass-through) and output data y (if provided).
        During fitting in a pipeline, y is provided and transformed.
        
        Parameters
        ----------
        X : array-like
            Input features (returned unchanged)
        y : array-like, default=None
            Target values to transform
            
        Returns
        -------
        X : array-like
            Original input features
        y_transformed : array-like or None
            Transformed target values (if y was provided)
        """
        # During pipeline.fit(), return X unchanged and transformed y
        if y is not None and self.methods is not None:
            y = np.asarray(y)
            methods = self.methods if isinstance(self.methods, list) else [self.methods]
            
            y_transformed = y.copy()
            for method in methods:
                if method in self.transformers_:
                    y_transformed = self.transformers_[method].transform(y_transformed)
            
            # Store the transformed y
            self.y_transformed_ = y_transformed
            
            return X, y_transformed
        
        # During pipeline.transform() or pipeline.predict(), return X unchanged
        return X
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : array-like
            Input features
        y : array-like, default=None
            Target values
            
        Returns
        -------
        X : array-like
            Original input features
        y_transformed : array-like or None
            Transformed target values (if y was provided)
        """
        return self.fit(X, y).transform(X, y)
    
    def inverse_transform_y(self, y_transformed):
        """
        Inverse transform the output data.
        
        Parameters
        ----------
        y_transformed : array-like
            Transformed target values
            
        Returns
        -------
        y : array-like
            Original target values
        """
        if self.methods is None or not self.transformers_:
            return y_transformed
            
        methods = self.methods if isinstance(self.methods, list) else [self.methods]
        y = y_transformed.copy()
        
        # Apply inverse transformations in reverse order
        for method in reversed(methods):
            if method in self.inverse_transformers_:
                y = self.inverse_transformers_[method](y)
                
        return y
    
    def inverse_transform(self, X):
        """
        Standard sklearn inverse_transform method.
        Takes transformed X (which in this case is actually the transformed y)
        and returns the original space representation.
        
        Parameters
        ----------
        X : array-like
            Transformed data
            
        Returns
        -------
        X_original : array-like
            Data reconstructed back to original space
        """
        # For pipeline compatibility, treat X as the transformed y values
        return self.inverse_transform_y(X)