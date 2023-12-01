import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np

data_dir = Path(__file__).parent.parent / "data"


def fetch_cardiac_data(dataset_name, train_test=False, test_size=0.2, random_state=42):
    """
    Fetches a dataset by name.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to fetch. Must be one of "atrial_cell", "four_chamber", or "circ_adapt".
        All data from here: https://zenodo.org/records/7405335
        "atrial_cell" is the ionic atrial cell model data from LH sampling.
        "four_chamber" is the 4-chamber model without LH sampling.
        "circ_adapt" is the circulatory adaptation model also from LH sampling.
    train_test : bool, optional
        If True, returns the dataset split into training and testing sets,
        X_train, X_test, y_train, y_test.
        If False, returns the entire dataset. Default is False.
    test_size : float, optional
        The proportion of the dataset to include in the test split.
        Only used if train_test is True. Default is 0.2.
    random_state : int, optional
        Controls the randomness of the dataset split.
        Only used if train_test is True. Default is 42.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Input data.
    y : ndarray, shape (n_samples, n_outputs)
        Output data.
    """

    data_dir_dataset = data_dir / dataset_name
    X = pd.read_csv(data_dir_dataset / "parameters.csv").to_numpy()
    y = pd.read_csv(data_dir_dataset / "outputs.csv").to_numpy()

    if train_test:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        return X, y


def fetch_cantilever_data(
    dataset_name, train_test=False, test_size=0.2, random_state=42
):
    """
    Fetches a dataset by name.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to fetch. Must be one of "cantilever" or "cantilever_truss".
        All data from Zack Conti (Turing).
        "cantilever": a dataset (996 dat points, input space is pseudo-randomly sampled) from
        the finite element analysis (frame) of a very simple cantilever
        beam where span of the beam, and it's cross-sectional depth are the inputs while
        maximum deflection and weight are the targets/outputs.
        "cantilever_truss": a more intricate cantilever truss design example whose
        geometry is parameterised by a set of geometric variables. In this dataset,
        deflection and weight are the targets/outputs while the remaining variables are
        inputs, controlling the geometry. Dataset size: 1000.
    train_test : bool, optional
        If True, returns the dataset split into training and testing sets,
        X_train, X_test, y_train, y_test.
        If False, returns the entire dataset. Default is False.
    test_size : float, optional
        The proportion of the dataset to include in the test split.
        Only used if train_test is True. Default is 0.2.
    random_state : int, optional
        Controls the randomness of the dataset split.
        Only used if train_test is True. Default is 42.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Input data.
    y : ndarray, shape (n_samples, n_outputs)
        Output data.
    """

    data_dir_dataset = data_dir / dataset_name
    X = pd.read_csv(data_dir_dataset / "parameters.csv").to_numpy()
    y = pd.read_csv(data_dir_dataset / "outputs.csv").to_numpy()

    if train_test:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        return X, y
