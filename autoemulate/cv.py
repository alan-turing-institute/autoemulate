from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from .types import Optional


# TODO: KFold seems to only accept a n_splits parameter of Int type so we should also enforce that here to avoid bugs
def kfold(folds: Optional[int] = None, shuffle: bool = True) -> KFold:
    """scikit-learn class for k-fold cross validation.

    Parameters
    ----------
    folds : int
        Number of folds. Must be at least 2.
    shuffle : bool
        Whether or not to shuffle the data before splitting.

    Returns
    -------
    kfold : sklearn.model_selection.KFold
        An instance of the KFold class.
    """
    return KFold(n_splits=folds, shuffle=shuffle)


CV_REGISTRY = {"kfold": kfold}
