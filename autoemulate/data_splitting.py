import numpy as np
from sklearn.model_selection import train_test_split

from .types import ArrayLike
from .types import Optional


def split_data(
    X: ArrayLike, test_size: float = 0.2, random_state: Optional[int] = None
) -> tuple[ArrayLike, ArrayLike]:
    """Splits the data into training and testing sets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Simulation input.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.

    Returns
    -------
    train_idx : array-like
        Indices of the training set.
    test_idx : array-like
        Indices of the testing set.
    """

    idxs = np.arange(X.shape[0])
    train_idxs, test_idxs = train_test_split(
        idxs, test_size=test_size, random_state=random_state
    )

    return train_idxs, test_idxs
