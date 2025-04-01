import numpy as np
from torch.utils.data import Dataset

from autoemulate.experimental.tuner import Tuner


def test_tuner():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    tuner = Tuner(dataset=Dataset(X, y), n_iter=10)
