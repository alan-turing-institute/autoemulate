from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = Path(__file__).parent.parent / "data"


def fetch_data(dataset, split=False, test_size=0.2, random_state=42):
    """
    Fetch a dataset by name.

    Parameters
    ----------
    dataset : str
        Dataset to load. Can be any of the following strings:

        - **cardiac1**: ionic atrial cell model data from LH sampling.
        - **cardiac2**: isotonic contraction ventricular cell model, no LH sampling.
        - **cardiac3**: CircAdapt: four-chamber pressure and volume CircAdapt ODE model from LH sampling.
        - **cardiac4**: four chamber: 3D-0D four-chamber electromechanics model to predict pressure and volume biomarkers for cardiac function.
        - **cardiac5**: passive mechanics: inflated volumes and mean atrial and ventricular fiber strains for a passive inflation.
        - **cardiac6**: tissue electrophysiology: predict total atrial and ventricular activation times with an Eikonal model.
        - **climate1**: GENIE model: predict climate variables SAT, ACC, VEGC, SOILC, MAXPMOC, OCN_O2, fCaCO3, SIAREA_S.
        - **engineering1**: Cantilever truss simulation.
    split : bool, optional
        Whether to split the data into training and testing sets. Default is False.
    test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.2.
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split. Default is 42.

    Returns
    -------
    X : array-like
        Simulation parameters / inputs.
    y : array-like
        Simulation outputs.
    """
    data_dir_dataset = data_dir / dataset / "processed"
    X = pd.read_csv(data_dir_dataset / "parameters.csv").to_numpy()
    y = pd.read_csv(data_dir_dataset / "outputs.csv").to_numpy()

    if split:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        return X, y
