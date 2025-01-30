import warnings

import numpy as np
import torch
from scipy.stats import loguniform
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from skorch.callbacks import GradientNormClipping
from skorch.callbacks import LRScheduler
from torch import nn

from autoemulate.emulators.neural_networks.cnp_module import CNPModule
from autoemulate.emulators.neural_networks.cnp_module_attn import AttnCNPModule
from autoemulate.emulators.neural_networks.datasets import cnp_collate_fn
from autoemulate.emulators.neural_networks.datasets import CNPDataset
from autoemulate.emulators.neural_networks.losses import CNPLoss
from autoemulate.utils import set_random_seed


class ConditionalNeuralProcess(RegressorMixin, BaseEstimator):
    """
    Conditional Neural Process (CNP) Regressor.

    This model integrates a meta-learning approach into a scikit-learn regressor. The fit() method accepts standard X, y inputs.
    A Conditional Neural Process (CNP) operates using context points and target points in episodes. For each episode, the model
    samples n_episode data points, a subset of which are designated as context points. All points from the episode, including
    the context points, are used as target points. Each episode thus represents a single sample in a meta-learning framework.
    During each epoch, we sample X.shape[0] times, ensuring that each data point is part of different contexts across episodes.
    The model can predict with a single input X, similar to other scikit-learn models. However, since the CNP requires context
    points for prediction, it stores the original X, y data as attributes, which are then used as context points for predicting
    new X data.

    Parameters
    ----------
    hidden_dim : int, default=64
        The number of hidden units in the neural network layers.
    latent_dim : int, default=64
        The dimensionality of the latent space.
    hidden_layers_enc : int, default=3
        The number of hidden layers in the encoder.
    hidden_layers_dec : int, default=3
        The number of hidden layers in the decoder.
    min_context_points : int, default=3
        The minimum number of context points to use during training.
    max_context_points : int, default=10
        The maximum number of context points to use during training.
    n_episode : int, default=32
        The number of episodes to sample during each epoch.
    max_epochs : int, default=100
        The maximum number of epochs to train the model.
    lr : float, default=0.01
        The learning rate for the optimizer.
    batch_size : int, default=16
        The number of samples per batch.
    activation : callable, default=torch.nn.ReLU
        The activation function to use in the neural network layers.
    optimizer : callable, default=torch.optim.AdamW
        The optimizer to use for training the model.
    device : str, default="cpu"
        The device to use for training. Options are "cpu" or "cuda".
    random_state : int, default=None
        The seed used by the random number generator.

    References
    ----------
    [1] Garnelo, M., Rosenbaum, D., Maddison, C., Ramalho, T., Saxton, D., Shanahan, M., Teh, Y.W., Rezende, D., & Eslami, S.M.A. (2018).
           Conditional Neural Processes. In International Conference on Machine Learning (pp. 1704-1713). PMLR.

    Attributes
    ----------
    input_dim_ : int
        The number of features in the input data.
    output_dim_ : int
        The number of targets in the output data.
    model_ : skorch.NeuralNetRegressor
        The neural network model used for regression.
    X_train_ : ndarray of shape (n_samples, n_features)
        The training input samples.
    y_train_ : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        The target values (real numbers) in the training set.

    Methods
    -------
    fit(X, y)
        Fit the model to the training data.
    predict(X, return_std=False)
        Predict using the trained model.

    Examples
    --------
    >>> import numpy as np
    >>> from autoemulate.emulators.cnp import ConditionalNeuralProcess
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.rand(100, 1)
    >>> cnp = ConditionalNeuralProcess(hidden_dim=128, latent_dim=128, hidden_layers=4, min_context_points=3, max_context_points=10, n_episode=32, max_epochs=100, lr=0.01, batch_size=16, activation=torch.nn.ReLU, optimizer=torch.optim.AdamW, device="cpu", random_state=42)
    >>> cnp.fit(X, y)
    >>> y_pred = cnp.predict(X)
    >>> y_pred.shape
    (100, 1)
    """

    def __init__(
        self,
        # architecture
        hidden_dim=64,
        latent_dim=64,
        hidden_layers_enc=3,
        hidden_layers_dec=3,
        # data per episode
        min_context_points=3,
        max_context_points=10,
        n_episode=32,
        # training
        max_epochs=100,
        lr=5e-3,
        batch_size=16,
        activation=nn.ReLU,
        optimizer=torch.optim.AdamW,
        normalize_y=True,
        # misc
        device="cpu",
        random_state=None,
        attention=False,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.hidden_layers_enc = hidden_layers_enc
        self.hidden_layers_dec = hidden_layers_dec
        self.min_context_points = min_context_points
        self.max_context_points = max_context_points
        self.n_episode = n_episode
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.activation = activation
        self.optimizer = optimizer
        self.normalize_y = normalize_y
        self.attention = attention
        self.device = device
        self.random_state = random_state
        if self.random_state is not None:
            set_random_seed(self.random_state)

    def fit(self, X, y):
        X, y = check_X_y(
            X,
            y,
            multi_output=True,
            dtype=np.float32,
            copy=True,
            ensure_2d=True,
            # ensure_min_samples=self.n_episode,
            y_numeric=True,
        )
        # y also needs to be float32 and 2d
        y = y.astype(np.float32)
        self.y_dim_ = y.ndim
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        self.input_dim_ = X.shape[1]
        self.output_dim_ = y.shape[1]

        # Normalize target value
        # the zero handler is from sklearn
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)

            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std

        if self.random_state is not None:
            set_random_seed(self.random_state)

        module = CNPModule if not self.attention else AttnCNPModule
        self.model_ = NeuralNetRegressor(
            module,
            module__input_dim=self.input_dim_,
            module__output_dim=self.output_dim_,
            module__hidden_dim=self.hidden_dim,
            module__latent_dim=self.latent_dim,
            module__hidden_layers_enc=self.hidden_layers_enc,
            module__hidden_layers_dec=self.hidden_layers_dec,
            module__activation=self.activation,
            dataset__min_context_points=self.min_context_points,
            dataset__max_context_points=self.max_context_points,
            dataset__n_episode=self.n_episode,
            max_epochs=self.max_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            device=self.device,
            dataset=CNPDataset,  # special dataset to sample context and target sets
            criterion=CNPLoss,
            iterator_train__collate_fn=cnp_collate_fn,  # special collate to different n in episodes
            iterator_valid__collate_fn=cnp_collate_fn,
            callbacks=[
                ("early_stopping", EarlyStopping(patience=10)),
                (
                    "lr_scheduler",
                    LRScheduler(policy="ReduceLROnPlateau", patience=5, factor=0.5),
                ),
                ("grad_norm", GradientNormClipping(gradient_clip_value=1.0)),
            ],
            # train_split=None,
            verbose=0,
        )
        self.model_.fit(X, y)
        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self)
        X = check_array(X, dtype=np.float32)
        X_context = torch.tensor(self.X_train_, dtype=torch.float32).unsqueeze(0)
        y_context = torch.tensor(self.y_train_, dtype=torch.float32).unsqueeze(0)
        X_target = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model_.module_.forward(X_context, y_context, X_target)

        # needs to be float64 to pass estimator tests
        # squeeze out batch dimension again so that score() etc. runs
        mean, logvar = predictions

        mean = mean[-X.shape[0] :].numpy().astype(np.float64).squeeze()
        logvar = logvar[-X.shape[0] :].numpy().astype(np.float64).squeeze()

        # undo normalization
        if self.normalize_y:
            mean = mean * self._y_train_std + self._y_train_mean
            var = np.exp(logvar) * (self._y_train_std**2)
            logvar = np.log(var)

        # make sure predictions have the same shape as y
        if self.y_dim_ == 1:
            mean = mean.ravel()
            logvar = logvar.ravel()
        else:
            mean = mean.reshape(-1, self.output_dim_)
            logvar = logvar.reshape(-1, self.output_dim_)

        if return_std:
            std = np.exp(0.5 * logvar)
            return mean, std
        else:
            return mean

    @staticmethod
    def get_grid_params(self, search_type="random"):
        if search_type == "random":
            param_space = {
                "max_epochs": [100, 200, 300],
                "batch_size": [16, 32],
                "hidden_dim": [32, 64, 128],
                "latent_dim": [32, 64, 128],
                "max_context_points": [5, 10, 15],
                "hidden_layers_enc": [2, 3, 4],
                "hidden_layers_dec": [2, 3, 4],
                "activation": [
                    nn.ReLU,
                    nn.GELU,
                ],
                "optimizer": [torch.optim.AdamW],  #
                "lr": loguniform(5e-4, 1e-3, 5e-3, 1e-2),
            }
        return param_space

    @property
    def model_name(self):
        if self.attention:
            return "AttentiveConditionalNeuralProcess"
        else:
            return "ConditionalNeuralProcess"

    def _more_tags(self):
        return {
            "multioutput": True,
            "poor_score": True,  # can be removed when max_epochs are ~1000 by default
            "non_deterministic": True,
        }

    def __repr__(self):
        # show architecture once fitted
        if hasattr(self, "model_"):
            return self.model_.__repr__()
        else:
            return super().__repr__()
