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

from autoemulate.emulators.conditional_neural_process import ConditionalNeuralProcess
from autoemulate.emulators.neural_networks.attn_cnp_module import AttnCNPModule
from autoemulate.emulators.neural_networks.cnp_module import CNPModule
from autoemulate.emulators.neural_networks.datasets import cnp_collate_fn
from autoemulate.emulators.neural_networks.datasets import CNPDataset
from autoemulate.emulators.neural_networks.losses import CNPLoss
from autoemulate.utils import set_random_seed


class AttentiveConditionalNeuralProcess(ConditionalNeuralProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        self.model_ = NeuralNetRegressor(
            AttnCNPModule,
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

    @property
    def model_name(self):
        return "AttentiveConditionalNeuralProcess"
