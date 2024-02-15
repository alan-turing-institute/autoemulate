import numpy as np

from autoemulate.data_splitting import split_data


def test_split_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    test_size = 0.2
    random_state = 42
    train_idxs, test_idxs = split_data(X, test_size, random_state)

    assert isinstance(train_idxs, np.ndarray)
    assert isinstance(test_idxs, np.ndarray)
    assert len(train_idxs) == 4
    assert len(test_idxs) == 1
