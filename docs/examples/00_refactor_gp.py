# %%
import torch
import numpy as np
import gpytorch
from autoemulate.emulators.gaussian_process import GaussianProcess
from sklearn.datasets import make_regression
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from autoemulate.refactor.gaussian_process import GaussianProcessRefactor
from autoemulate.refactor.utils import sample_data_y1d


# %%
train_x, train_y = sample_data_y1d()
gp = GaussianProcessRefactor(train_x, train_y, normalize_y=False)
gp.fit(train_x, train_y)

# %%
with torch.no_grad():
    y_pred = gp.predict(train_x).sample().numpy()
    y_pred_mean = gp.predict(train_x).mean.numpy()
    y_pred_std = gp.predict(train_x).sample_n(1000).std(0).numpy()

# %%
plt.scatter(train_x[:, 0], train_y)
plt.scatter(train_x[:, 0], y_pred_mean)
plt.fill_between(
    train_x[:, 0],
    (y_pred_mean - y_pred_std).flatten(),
    (y_pred_mean + y_pred_std).flatten(),
)
plt.show()

# %%
# %%

# %%
