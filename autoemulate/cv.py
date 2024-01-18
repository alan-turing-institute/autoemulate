from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


def kfold(folds=None, shuffle=True):
    """scikit-learn class for k-fold cross validation.

    Parameters
    ----------
    folds : int
            Number of folds.
    shuffle : bool
            Whether or not to shuffle the data before splitting.

    Returns
    -------
    kfold : sklearn.model_selection.KFold
            An instance of the KFold class.
    """
    return KFold(n_splits=folds, shuffle=shuffle)


CV_REGISTRY = {"kfold": kfold}
