import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from autoemulate.preprocess_target import get_dim_reducer
from autoemulate.preprocess_target import NoChangeTransformer
from autoemulate.preprocess_target import NonTrainableTransformer
from autoemulate.preprocess_target import TargetPCA
from autoemulate.preprocess_target import VAE
from autoemulate.preprocess_target import TargetVAE

# Import the components to test

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    X, _ = make_blobs(n_samples=100, n_features=10, centers=3, random_state=42)
    return X


@pytest.fixture
def sample_data_1d():
    """Generate 1D sample data for testing."""
    return np.random.randn(100, 1)


def test_get_dim_reducer():
    """Test the dim reducer factory function."""
    # Test PCA
    pca = get_dim_reducer("PCA", reduced_dim=3)
    assert isinstance(pca, PCA)
    assert pca.n_components == 3

    # Test VAE
    vae = get_dim_reducer("VAE", reduced_dim=3)
    assert isinstance(vae, TargetVAE)
    assert vae.latent_dim == 3

    # Test None case
    none_transformer = get_dim_reducer("None")
    assert isinstance(none_transformer, NoChangeTransformer)

    # Test invalid case
    with pytest.raises(ValueError):
        get_dim_reducer("invalid")


def test_target_pca(sample_data, sample_data_1d):
    """Test TargetPCA functionality."""
    # Test initialization
    tpca = TargetPCA(n_components=3)
    assert tpca.n_components == 3

    # Test fit/transform with multi-dimensional data
    X = sample_data
    X_trans = tpca.fit_transform(X)
    assert X_trans.shape == (X.shape[0], 3)

    # Test inverse transform
    X_reconstructed = tpca.inverse_transform(X_trans)
    assert X_reconstructed.shape == X.shape

    # Test with 1D data
    X_1d = sample_data_1d
    tpca_1d = TargetPCA(n_components=1)
    X_1d_trans = tpca_1d.fit_transform(X_1d)
    assert X_1d_trans.shape == (X_1d.shape[0], 1)

    # Test inverse_transform_std
    x_std = np.random.rand(10, 3)  # 10 samples, 3 components
    std_transformed = tpca.inverse_transform_std(x_std)
    assert std_transformed.shape == (10, X.shape[1])

    # Test validation
    with pytest.raises(ValueError):
        tpca.transform(None)


def test_vae_output_preprocessor(sample_data):
    """Test VAEOutputPreprocessor functionality."""
    # Test initialization
    vae = TargetVAE(latent_dim=3, epochs=5, verbose=False)
    assert vae.latent_dim == 3
    assert not vae.is_fitted_

    # Test fit/transform
    X = sample_data
    vae.fit(X)
    assert vae.is_fitted_
    assert vae.input_dim == X.shape[1]

    X_trans = vae.transform(X)
    assert X_trans.shape == (X.shape[0], 3)

    # Test inverse transform
    X_reconstructed = vae.inverse_transform(X_trans)
    assert X_reconstructed.shape == X.shape

    # Test fit_transform
    X_trans2 = vae.fit_transform(X)
    assert X_trans2.shape == (X.shape[0], 3)

    # Test not fitted error
    vae2 = TargetVAE(latent_dim=3)
    with pytest.raises(ValueError):
        vae2.transform(X)


def test_vae_model():
    """Test the VAE model architecture and forward pass."""
    # Test initialization
    vae = VAE(input_dim=10, hidden_layers=[8, 6], latent_dim=3)
    assert isinstance(vae.encoder, nn.Sequential)
    assert isinstance(vae.decoder, nn.Sequential)

    # Test forward pass
    x = torch.randn(5, 10)  # batch of 5 samples, 10 features
    recon, mu, log_var = vae(x)
    assert recon.shape == x.shape
    assert mu.shape == (5, 3)
    assert log_var.shape == (5, 3)

    # Test encode/decode
    mu, log_var = vae.encode(x)
    assert mu.shape == (5, 3)
    z = vae.reparameterize(mu, log_var)
    assert z.shape == (5, 3)
    recon = vae.decode(z)
    assert recon.shape == x.shape

    # Test loss function
    loss = vae.loss_function(recon, x, mu, log_var)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar


def test_no_change_transformer(sample_data):
    """Test the NoChangeTransformer."""
    transformer = NoChangeTransformer()

    X = sample_data
    X_trans = transformer.fit_transform(X)
    assert np.array_equal(X, X_trans)

    X_inv = transformer.inverse_transform(X)
    assert np.array_equal(X, X_inv)

    x_std = np.random.rand(10, 3)
    std_inv = transformer.inverse_transform_std(x_std)
    assert np.array_equal(x_std, std_inv)


def test_non_trainable_transformer(sample_data):
    """Test the non_trainable_transformer wrapper."""
    # Wrap a PCA
    pca = TargetPCA(n_components=3)
    pca.fit(sample_data)
    transformer = NonTrainableTransformer(pca)

    X = sample_data
    X_trans = transformer.transform(X)
    assert X_trans.shape == (X.shape[0], 3)

    # Test inverse transform
    X_inv = transformer.inverse_transform(X_trans)
    assert X_inv.shape == X.shape

    # Test inverse_transform_std
    x_std = np.random.rand(10, 3)
    std_inv = transformer.inverse_transform_std(x_std)
    assert std_inv.shape == (10, X.shape[1])


@pytest.mark.parametrize("reducer_type", ["PCA", "VAE", "None"])
def test_reducer_in_pipeline(sample_data, reducer_type):
    """Test that reducers work in sklearn pipelines."""
    X = sample_data
    reducer = get_dim_reducer(reducer_type, reduced_dim=3)

    pipeline = Pipeline([("scaler", StandardScaler()), ("reducer", reducer)])

    # Test fit/transform
    X_trans = pipeline.fit_transform(X)
    if reducer_type == "None":
        assert X_trans.shape == X.shape
    else:
        assert X_trans.shape == (X.shape[0], 3)

    # Test inverse transform if available
    if hasattr(pipeline, "inverse_transform"):
        X_inv = pipeline.inverse_transform(X_trans)
        assert X_inv.shape == X.shape
