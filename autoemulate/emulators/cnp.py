import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback

from autoemulate.emulators.neural_networks.cnp_module import CNPModule
from autoemulate.emulators.neural_networks.cnp_module import GaussianNLLLoss


class CNP(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_dim=64,
        latent_dim=64,
        context_points=5,
        max_epochs=100,
        lr=0.001,
        batch_size=32,
        device="cpu",
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.context_points = context_points
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self.input_dim_ = X.shape[1]
        self.output_dim_ = y.shape[1] if len(y.shape) > 1 else 1
        self.model_ = NeuralNetRegressor(
            CNPModule,
            module__input_dim=self.input_dim_,
            module__output_dim=self.output_dim_,
            module__hidden_dim=self.hidden_dim,
            module__latent_dim=self.latent_dim,
            module__context_points=self.context_points,
            max_epochs=self.max_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            device=self.device,
            criterion=GaussianNLLLoss,
        )
        X_dict = {"X": X, "y": y}
        self.model_.fit(X_dict, y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self)
        X = check_array(X)

        X = torch.tensor(X, dtype=torch.float32)

        X_dict = {
            "X": torch.cat([torch.tensor(self.X_train_, dtype=torch.float32), X]),
            "y": torch.cat(
                [
                    torch.tensor(self.y_train_, dtype=torch.float32),
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

    @property
    def model_name(self):
        return self.module_name

    def _more_tags(self):
        return {
            "multioutput": True,
            "poor_score": True,
            "_xfail_checks": {
                "check_no_attributes_set_in_init": "skorch initialize attributes in __init__.",
                "check_regressors_no_decision_function": "skorch NeuralNetRegressor class implements the predict_proba.",
                "check_parameters_default_constructible": "skorch NeuralNet class callbacks parameter expects a list of callables.",
                "check_dont_overwrite_parameters": "the change of public attribute module__input_size is needed to support dynamic input size.",
                "check_estimators_overwrite_params": "in order to support dynamic input and output size, we have to overwrite module__input_size and module__output_size during fit.",
                "check_estimators_empty_data_messages": "the error message cannot print module__input_size the module has not been initialized",
                "check_set_params": "_params_to_validate must be a list or set, while check_set_params set it to a float which causes AttributeError",
            },
        }
