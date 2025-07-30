from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from autoemulate.core.types import TensorLike

data_dir = Path(__file__).parent.parent.parent / "data"


def fetch_data(
    dataset: str, split: bool = False, test_size: float = 0.2, random_state: int = 42
) -> list[TensorLike]:
    """
    Fetch a dataset by name.

    Parameters
    ----------
    dataset: str
        Dataset to load. Can be any of the following strings:
        - **cardiac1**: ionic atrial cell model data from LH sampling.
        - **cardiac2**: isotonic contraction ventricular cell model, no LH sampling.
        - **cardiac3**: CircAdapt: four-chamber pressure and volume CircAdapt ODE model
                        from LH sampling.
        - **cardiac4**: four chamber: 3D-0D four-chamber electromechanics model to
                        predict pressure and volume biomarkers for cardiac function.
        - **cardiac5**: passive mechanics: inflated volumes and mean atrial and
                        ventricular fiber strains for a passive inflation.
        - **cardiac6**: tissue electrophysiology: predict total atrial and ventricular
                        activation times with an Eikonal model.
        - **climate1**: GENIE model: predict climate variables SAT, ACC, VEGC, SOILC,
                        MAXPMOC, OCN_O2, fCaCO3, SIAREA_S.
        - **engineering1**: Cantilever truss simulation.
        - **reactiondiffusion1**: Reaction-diffusion system simulated data. Spatial
                        snapshots in correspondence for different values of reaction
                        and  ddiffusion parameters. (50 snapshots in correspondence of
                        50 different parameter sets at time T=10).
    split: bool
        Whether to split the data into training and testing sets. Default is False.
    test_size: float
        The proportion of the dataset to include in the test split. Default is 0.2.
    random_state: int
        Controls the shuffling applied to the data before applying the split. Default
        is 42.

    Returns
    -------
    list[TensorLike]
        Tensors of simulation parameters / inputs and simulation outputs. If `split`
        is True, returns [x_train, x_test, y_train, y_test].
    """
    data_dir_dataset = data_dir / dataset / "processed"
    X = pd.read_csv(data_dir_dataset / "parameters.csv").to_numpy()
    Y = pd.read_csv(data_dir_dataset / "outputs.csv").to_numpy()

    if split:
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
        return [
            torch.tensor(x_train),
            torch.tensor(x_test),
            torch.tensor(y_train),
            torch.tensor(y_test),
        ]
    return [torch.tensor(X), torch.tensor(Y)]
