import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin

class VariationalAutoencoder(nn.Module):
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
    def __init__(self, input_dim, hidden_dims, latent_dim, verbose=False):
        super(VariationalAutoencoder, self).__init__()
        
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

    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        """
        Calculate VAE loss: reconstruction + KL divergence.
        
        Parameters
        ----------
        recon_x : torch.Tensor
            Reconstructed input
        x : torch.Tensor
            Original input
        mu : torch.Tensor
            Mean of latent distribution
        log_var : torch.Tensor
            Log variance of latent distribution
        beta : float, default=1.0
            Weight for KL divergence term (beta-VAE)
            
        Returns
        -------
        total_loss : torch.Tensor
            Combined reconstruction and KL divergence loss
        """
        # MSE reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + beta * kl_loss

class VariationalAutoencoderDimReducer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper for PyTorch Variational Autoencoder to use as dimensionality reducer.
    Implements fit, transform, and inverse_transform methods required for sklearn pipelines.
    
    Parameters
    ----------
    encoding_dim : int, default=10
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
    
    def __init__(self, encoding_dim=10, hidden_layers=None, epochs=100, batch_size=32, 
                 learning_rate=0.001, beta=1.0, device=None, random_state=None, verbose=False):
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers if hidden_layers is not None else [32, 16]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        self.verbose = verbose
        
        # Will be initialized during fit
        self.model = None
        self.input_dim = None
        self.is_fitted_ = False
        
    def _check_is_fitted(self):
        """Check if the model is fitted."""
        if not self.is_fitted_:
            raise ValueError("This VariationalAutoencoderDimReducer instance is not fitted yet. "
                            "Call 'fit' with appropriate arguments before using this estimator.")
    
    def fit(self, X, y=None):
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
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize the model
        self.model = VariationalAutoencoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_layers,
            latent_dim=self.encoding_dim
        ).to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                # Forward pass
                recon_batch, mu, log_var = self.model(batch_x)
                
                # Calculate loss
                loss = self.model.loss_function(recon_batch, batch_x, mu, log_var, beta=self.beta)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print training progress
            #if self.verbose and (epoch + 1) % 10 == 0:
            #    print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.6f}')
        
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
        
        # Create data loader
        batch_size = min(self.batch_size, len(X))
        dataloader = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)
        
        # Encode data
        encoded_data = []
        self.model.eval()
        with torch.no_grad():
            for batch_x in dataloader:
                mu, _ = self.model.encode(batch_x)
                encoded_data.append(mu.cpu().numpy())
        
        return np.vstack(encoded_data)
    
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
        
        # Create data loader
        batch_size = min(self.batch_size, len(X))
        dataloader = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)
        
        # Decode data
        decoded_data = []
        self.model.eval()
        with torch.no_grad():
            for batch_z in dataloader:
                decoded = self.model.decode(batch_z)
                decoded_data.append(decoded.cpu().numpy())
        
        return np.vstack(decoded_data)
    
    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored. Present for API consistency.
            
        Returns
        -------
        X_new : ndarray of shape (n_samples, encoding_dim)
            Transformed array.
        """
        # Fit the model
        self.fit(X, y)
        # Return the transformed data
        return self.transform(X)
    
    def sample(self, n_samples=1):
        """
        Generate samples from the latent space.
        
        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate
            
        Returns
        -------
        samples : ndarray of shape (n_samples, n_features)
            Generated samples
        """
        self._check_is_fitted()
        
        self.model.eval()
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(n_samples, self.encoding_dim).to(self.device)
            # Decode samples
            samples = self.model.decode(z)
            
        return samples.cpu().numpy()
    
    

class Autoencoder(nn.Module):
    """PyTorch Autoencoder neural network architecture."""
    
    def __init__(self, input_dim, encoding_dim, hidden_layers=None, verbose=False):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of the input data
        encoding_dim : int
            Dimension of the encoded representation
        hidden_layers : list of int, default=None
            List of hidden layer sizes for encoder (decoder will be symmetric)
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.verbose = verbose
        
        # Default architecture if no hidden layers specified
        if hidden_layers is None:
            hidden_layers = [input_dim // 2]
        
        # Build encoder layers
        encoder_layers = []
        last_size = input_dim
        
        # Add hidden layers
        for h_dim in hidden_layers:
            encoder_layers.append(nn.Linear(last_size, h_dim))
            encoder_layers.append(nn.ReLU())
            last_size = h_dim
        
        # Add bottleneck layer
        encoder_layers.append(nn.Linear(last_size, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder layers (mirroring the encoder)
        decoder_layers = []
        last_size = encoding_dim
        
        # Add hidden layers (in reverse)
        for h_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(last_size, h_dim))
            decoder_layers.append(nn.ReLU())
            last_size = h_dim
        
        # Add output layer
        decoder_layers.append(nn.Linear(last_size, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode the input"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode the latent representation"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through the autoencoder"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


class AutoencoderDimReducer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper for PyTorch Autoencoder to use as dimensionality reducer.
    Implements fit, transform, and inverse_transform methods required for sklearn pipelines.
    """
    
    def __init__(self, encoding_dim=10, hidden_layers=None, epochs=100, batch_size=32, 
                 learning_rate=0.001, device=None, random_state=None, verbose=False):
        """
        Parameters
        ----------
        encoding_dim : int, default=10
            Dimension of the encoded representation
        hidden_layers : list of int, default=None
            List of hidden layer sizes for encoder
        epochs : int, default=100
            Number of training epochs
        batch_size : int, default=32
            Batch size for training
        learning_rate : float, default=0.001
            Learning rate for optimizer
        device : str, default=None
            Device to run the model on ('cpu' or 'cuda'). If None, use cuda if available.
        random_state : int, default=None
            Random seed for reproducibility
        """
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        self.verbose = verbose
        
        # Will be initialized during fit
        self.model = None
        self.input_dim = None
        self.is_fitted_ = False
    
    def _check_is_fitted(self):
        """Check if the model is fitted."""
        if not self.is_fitted_:
            raise ValueError("This AutoencoderDimReducer instance is not fitted yet. "
                            "Call 'fit' with appropriate arguments before using this estimator.")
    
    def _get_data_loader(self, X, shuffle=True):
        """Create a DataLoader for the given data."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)  # Input and target are the same
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def fit(self, X, y=None):
        """
        Fit the autoencoder on the training data.
        
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
        
        # Initialize the model
        self.model = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            hidden_layers=self.hidden_layers
        ).to(self.device)
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Create data loader
        train_loader = self._get_data_loader(X)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in train_loader:
                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_x)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Optional: print training progress
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(train_loader):.6f}')
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Reduce dimensionality by encoding the data.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to transform.
            
        Returns
        -------
        X_new : ndarray of shape (n_samples, encoding_dim)
            Transformed array.
        """
        self._check_is_fitted()
        
        # Create data tensor and loader
        data_loader = self._get_data_loader(X, shuffle=False)
        
        # Encode data
        encoded_data = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, _ in data_loader:
                encoded = self.model.encode(batch_x)
                encoded_data.append(encoded.cpu().numpy())
        
        return np.vstack(encoded_data)
    
    def inverse_transform(self, X):
        """
        Transform encoded data back to the original space.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, encoding_dim)
            The encoded data.
            
        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Data in original space.
        """
        self._check_is_fitted()
        
        # Create data tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        data_loader = torch.utils.data.DataLoader(X_tensor, batch_size=self.batch_size, shuffle=False)
        
        # Decode data
        decoded_data = []
        self.model.eval()
        with torch.no_grad():
            for batch_z in data_loader:
                decoded = self.model.decode(batch_z)
                decoded_data.append(decoded.cpu().numpy())
        
        return np.vstack(decoded_data)
    
    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored. Present for API consistency.
            
        Returns
        -------
        X_new : ndarray of shape (n_samples, encoding_dim)
            Transformed array.
        """
        # Fit the model
        self.fit(X)
        # Return the transformed data
        return self.transform(X)
    

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