import numpy as np
import torch
from scipy.stats import loguniform
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from skopt.space import Real
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback

from autoemulate.emulators.neural_networks.cnp_module import CNPModule
from autoemulate.emulators.neural_networks.cnp_module import RobustGaussianNLLLoss
from autoemulate.utils import set_random_seed


class ConditionalNeuralProcess(RegressorMixin, BaseEstimator):
    """
    Conditional Neural Process (CNP) Regressor.

    Parameters
    ----------
    hidden_dim : int, default=64
        The number of hidden units in the neural network layers.
    latent_dim : int, default=64
        The dimensionality of the latent space.
    context_points : int, default=16
        The number of context points to use during training.
    max_epochs : int, default=100
        The maximum number of epochs to train the model.
    lr : float, default=0.001
        The learning rate for the optimizer.
    batch_size : int, default=32
        The number of samples per batch.
    device : str, default="cpu"
        The device to use for training. Options are "cpu" or "cuda".

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
    >>> cnp = ConditionalNeuralProcess(hidden_dim=32, latent_dim=32, context_points=10, max_epochs=50, lr=0.01, batch_size=16, device="cpu")
    >>> cnp.fit(X, y)
    >>> y_pred = cnp.predict(X)
    >>> y_pred.shape
    (100, 1)
    """

    def __init__(
        self,
        hidden_dim=64,
        latent_dim=64,
        context_proportion=0.5,
        max_epochs=100,
        lr=1e-3,
        batch_size=32,
        device="cpu",
        **kwargs,
    ):
        if "random_state" in kwargs:
            setattr(self, "random_state", kwargs.pop("random_state"))
            set_random_seed(self.random_state)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.context_proportion = context_proportion
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y):
        X, y = check_X_y(
            X, y, multi_output=True, y_numeric=True, dtype=np.float32, copy=True
        )
        y = y.astype(np.float32)
        # convert y to 2d if its 1d
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.input_dim_ = X.shape[1]
        self.output_dim_ = y.shape[1] if len(y.shape) > 1 else 1
        self.model_ = NeuralNetRegressor(
            CNPModule,
            module__input_dim=self.input_dim_,
            module__output_dim=self.output_dim_,
            module__hidden_dim=self.hidden_dim,
            module__latent_dim=self.latent_dim,
            module__context_proportion=self.context_proportion,
            max_epochs=self.max_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            device=self.device,
            criterion=RobustGaussianNLLLoss,
            train_split=None,
            verbose=1,
        )
        X_dict = {"X": X, "y": y}
        self.model_.fit(X_dict, y)
        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self)
        X = check_array(X, dtype=np.float32)

        X_dict = {
            "X": torch.cat([torch.from_numpy(self.X_train_), torch.from_numpy(X)]),
            "y": torch.cat(
                [
                    torch.from_numpy(self.y_train_),
                    torch.zeros((X.shape[0], self.output_dim_), dtype=torch.float32),
                ]
            ),
        }

        with torch.no_grad():
            predictions = self.model_.forward(X_dict)

        # Extract predictions for new data points
        mean, logvar = predictions
        mean = mean[-X.shape[0] :].numpy()
        logvar = logvar[-X.shape[0] :].numpy()

        if return_std:
            std = np.exp(0.5 * logvar)
            return mean, std
        else:
            return mean

    @staticmethod
    def get_grid_params(search_type: str = "random"):
        param_space = {
            "max_epochs": np.arange(10, 500, 10).tolist(),
            "batch_size": np.arange(2, 128, 2).tolist(),
            # "module__hidden_layers": np.arange(1, 4).tolist(),
            # "module__hidden_dim": np.arange(50, 500, 50).tolist(),
            # "module__latent_dim": np.arange(50, 500, 50).tolist(),
            # "module__hidden_activation": [
            #     nn.ReLU,
            #     nn.Tanh,
            #     nn.Sigmoid,
            #     nn.GELU,
            # ],
            # "optimizer": [torch.optim.AdamW, torch.optim.LBFGS, torch.optim.SGD],  #
            # "optimizer__weight_decay": (1 / 10 ** np.arange(1, 9)).tolist(),
        }
        match search_type:
            case "random":
                param_space |= {
                    "lr": loguniform(1e-6, 1e-4),
                }
            case "bayes":
                param_space |= {
                    # "optimizer": Categorical(param_space["optimizer"]),
                    "lr": Real(1e-6, 1e-4, prior="log-uniform"),
                }
            case _:
                raise ValueError(f"Invalid search type: {search_type}")

        return param_space

    @property
    def model_name(self):
        return self.__class__.__name__

    def _more_tags(self):
        return {
            "multioutput": True,
            "poor_score": True,
            # "_xfail_checks": {
            #     "check_no_attributes_set_in_init": "skorch initialize attributes in __init__.",
            #     "check_regressors_no_decision_function": "skorch NeuralNetRegressor class implements the predict_proba.",
            #     "check_parameters_default_constructible": "skorch NeuralNet class callbacks parameter expects a list of callables.",
            #     "check_dont_overwrite_parameters": "the change of public attribute module__input_size is needed to support dynamic input size.",
            #     "check_estimators_overwrite_params": "in order to support dynamic input and output size, we have to overwrite module__input_size and module__output_size during fit.",
            #     "check_estimators_empty_data_messages": "the error message cannot print module__input_size the module has not been initialized",
            #     "check_set_params": "_params_to_validate must be a list or set, while check_set_params set it to a float which causes AttributeError",
            # },
        }
