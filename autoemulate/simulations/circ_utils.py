# Started by importing the ModularCirc package.
# * Imported Naghavi model and its parameters from ModularCirc models submodule
# * Imported the solver from ModularCirc solver submodule
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ModularCirc.Analysis.BaseAnalysis import BaseAnalysis
from ModularCirc.Models.NaghaviModel import NaghaviModel
from ModularCirc.Models.NaghaviModel import NaghaviModelParameters
from ModularCirc.Solver import Solver
from tqdm import tqdm


def signal_get_pulse(signal, dt, num=100):
    """
    Function for resampling the raw signal to a standard resolution

    Args:
            signal (array): the raw signal generated by the model
            dt (float): the raw signal resolution
            num (int, optional): the new number of s. Defaults to 100.

    Returns:
            _type_: _description_
    """

    ind = np.argmin(signal)
    ncycle = len(signal)
    new_signal = np.interp(
        np.linspace(0, ncycle, num), np.arange(ncycle), np.roll(signal, -ind)
    )
    new_dt = ncycle / (num - 1) * dt
    return new_dt, new_signal


def run_case(row, output_path, N_cycles, dt):
    """
    Function for running the simulation using a specific combination of parameters.

    Args:
        row (pd.Series): a row in a DataFrame describing a specific combination of parameters
        output_path (str): the directory into which the results of the simulation are saved
        N_cycles (int): the maximum number of heartbeats used to run a simulation
        dt (float): the time step size
    """

    TEMPLATE_TIME_SETUP_DICT = {
        "name": "TimeTest",
        "ncycles": N_cycles,  # 20
        "tcycle": row["T"],
        "dt": dt,
        "export_min": 1,
    }

    ## Define the parameter instance..
    parobj = (
        NaghaviModelParameters()
    )  # replace the .. with the correct Class and inputs if applicable
    for key, val in row.items():
        if key == "T":
            continue
        obj, param = key.split(".")
        parobj._set_comp(
            obj,
            [
                obj,
            ],
            **{param: val}
        )

    # Define the model instance..
    model = NaghaviModel(
        time_setup_dict=TEMPLATE_TIME_SETUP_DICT, parobj=parobj, suppress_printing=True
    )  # replace the ... with the correct Class and inputs if applicable

    ## Define the solver instance..
    solver = Solver(
        model=model
    )  # replace the .. with the correct Class and inputs if applicable

    solver.setup(
        suppress_output=True,
        optimize_secondary_sv=False,
        conv_cols=["p_ao"],
        method="LSODA",
    )
    solver.solve()

    # If the solver hasn't converged we quit the function early..
    if not solver.converged:
        return False

    # BaseAnalysis is a utility class used to post-process the outputs from the model..
    analysis = BaseAnalysis(model)
    analysis.compute_cardiac_output("lv")
    ind = model.time_object.n_c

    # Get the raw pressure signal...
    raw_signal = model.components["ao"].P[-ind:]

    # Resamples the raw signals (variable length) to a standard 100 time points resolution
    resampled_dt, resampled_signal = signal_get_pulse(raw_signal, model.time_object.dt)

    # Create a pd.Series to store the post-processed version of the signal..
    series_rs = pd.Series(resampled_signal)
    series_rs["C0 (mm^3/s)"] = analysis.CO
    series_rs["dt"] = resampled_dt
    series_rs.index.name = "Index"

    # Save the results of the simulations..
    series_rs.to_csv(os.path.join(output_path, "ao" + str(row.name) + ".csv"))

    return True


def run_in_parallel(
    output_directory: str, N_cycles: int, dt: float, parameter_data_frame: pd.DataFrame
):
    successful_runs = joblib.Parallel(n_jobs=5)(
        joblib.delayed(run_case)(row, output_directory, N_cycles, dt)
        for _, row in tqdm(
            parameter_data_frame.iterrows(), total=len(parameter_data_frame)
        )
    )
    print("done")

    return successful_runs


def simulation_loader(input_directory):
    file_list = os.listdir(input_directory)
    file_list_clean = [file for file in file_list if file.split(".")[-1] == "csv"]

    file_series = pd.DataFrame(
        file_list_clean,
        columns=[
            "file",
        ],
    )
    file_series["Index"] = [int(file[2:].split(".")[0]) for file in file_list_clean]

    file_series.sort_values("Index", inplace=True)
    file_series.set_index("Index", inplace=True)

    # Define a dataframe for the values collected from the simulation...
    signal_df = file_series.apply(
        lambda row: list(
            pd.read_csv(os.path.join(input_directory, row["file"]), index_col="Index")
            .to_numpy()
            .reshape((-1))
        ),
        axis=1,
        result_type="expand",
    )
    template_columns = pd.read_csv(
        os.path.join(input_directory, file_series.iloc[0, 0])
    )["Index"].to_dict()
    signal_df.rename(columns=template_columns, inplace=True)

    return signal_df


dict_parameters_condensed_range = dict()
dict_parameters_condensed_single = dict()


def condense_dict_parameters(dict_param: dict, prev=""):
    for key, val in dict_param.items():
        if len(prev) > 0:
            new_key = prev.split(".")[-1] + "." + key
        else:
            new_key = key
        if isinstance(val, dict):
            condense_dict_parameters(val, new_key)
        else:
            if len(val) > 1:
                value, r = val
                dict_parameters_condensed_range[new_key] = tuple(np.array(r) * value)
            else:
                dict_parameters_condensed_single[new_key] = val[0]
    return


######## DEFINED A FUNCTION TO PLOT THE VARIANCE
def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    axs[0].bar(grid, explained_variance_ratio, log=True)
    axs[0].set(xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0))

    # Cumulative Variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    axs[1].semilogy(grid, cumulative_explained_variance, "o-")
    axs[1].set(
        xlabel="Component",
        title="% Cumulative Variance",
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    fig.tight_layout()
    return axs


def get_component(X, component_matrix, component_id, scaler):
    return scaler.inverse_transform(
        X
        @ component_matrix[component_id, :].T.reshape((-1, 1))
        * component_matrix[component_id, :].reshape((1, -1))
    )


def scale_time_parameters_and_asign_to_components(df):
    # Scale the time parameters down based on specific pulse duration
    # 800 ms in this case

    df["la.delay"] = df["la.delay"] * df["T"] / 800.0

    df["la.t_tr"] = df["la.t_tr"] * df["T"] / 800.0
    df["lv.t_tr"] = df["lv.t_tr"] * df["T"] / 800.0

    df["la.tau"] = df["la.tau"] * df["T"] / 800.0
    df["lv.tau"] = df["lv.tau"] * df["T"] / 800.0

    df["la.t_max"] = df["la.t_max"] * df["T"] / 800.0
    df["lv.t_max"] = df["lv.t_max"] * df["T"] / 800.0
    return
