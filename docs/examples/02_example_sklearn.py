# %%
from sklearn.linear_model import LinearRegression
import numpy as np

from autoemulate.refactor.base import SklearnBackend

X_train = np.random.rand(100, 3)
y_train = np.random.rand(100)

model = SklearnBackend(LinearRegression())  # Wrap any sklearn model
model.fit(X_train, y_train)  # API is unchanged
model.cross_validate((X_train, y_train))
