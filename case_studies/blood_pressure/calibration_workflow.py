import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from cardiac_simulator import NaghaviSimulator
from matplotlib.colors import Normalize

from autoemulate.calibration.bayes import BayesianCalibration
from autoemulate.calibration.history_matching import HistoryMatchingWorkflow
from autoemulate.core.compare import AutoEmulate
from autoemulate.core.sensitivity_analysis import SensitivityAnalysis
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators.gaussian_process.kernel import matern_3_2_kernel
from autoemulate.emulators.nn.mlp import MLP

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


def plot_implausibility(
    df, wave, top_parameters_sa, ref_val, output_path="implausibility_plot"
):
    """
    Plots the implausibility scores for a given wave and saves the figure.

    Parameters:
    - df: DataFrame containing the results for the wave.
    - wave: The wave number to filter the DataFrame.
    - top_parameters_sa: List of top parameters for sensitivity analysis.
    - ref_val: List of reference values for the parameters.
    - output_path: Path to save the generated plot.
    """
    df = df[df["Wave"] == wave]

    g = sns.PairGrid(df, vars=top_parameters_sa, corner=True)

    # Normalization + colormap for continuous values
    norm = Normalize(vmin=df["Implausibility"].min(), vmax=df["Implausibility"].max())
    cmap = plt.cm.viridis

    def scatter_continuous(x, y, color=None, **kwargs):
        ax = plt.gca()
        sc = ax.scatter(
            x,
            y,
            c=df.loc[x.index, "Implausibility"],
            cmap=cmap,
            norm=norm,
            s=15,
            alpha=0.7,
        )
        return sc

    g.map_lower(scatter_continuous)
    g.map_diag(sns.histplot, kde=False, color="gray")

    # Add reference points
    for i, xi in enumerate(top_parameters_sa):
        for j, yj in enumerate(top_parameters_sa):
            if j < i:  # lower triangle only
                ax = g.axes[i, j]
                ax.scatter(
                    ref_val[j],
                    ref_val[i],
                    color="white",
                    s=60,
                    edgecolor="black",
                    marker="X",
                    zorder=5,
                    label=(
                        "True value"
                        if (i == len(top_parameters_sa) - 1 and j == 0)
                        else None
                    ),
                )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gcf().axes, shrink=0.7, label="Implausibility")

    # Global legend (handles all subplots)
    handles, labels = g.axes[-1, 0].get_legend_handles_labels()
    g.fig.legend(handles, labels, loc="upper right", frameon=True)
    g.fig.suptitle(f"Results for Wave {wave}", fontsize=16)

    # Save the figure
    plt.savefig(f"{output_path}_{wave}.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    seed = 42
    set_random_seed(seed)

    n_waves = 5
    n_parameters = 2

    # #### Set up simulator and generate data
    #

    simulator = NaghaviSimulator(
        output_variables=["lv.P"],  # We simulate the left ventricle pressure
        n_cycles=300,
        dt=0.001,
    )

    # The simulator comes with predefined input parameters ranges.

    # In[3]:

    # We can sample from those using Latin Hypercube Sampling to generate data to train the emulator with.

    # In[4]:

    N_samples = 1024

    # We can now use the simulator to generate predictions for the sampled parameters. Alternatively, for convenience. we can load already simulated data.

    # In[5]:

    save = True

    if not os.path.exists(f"simulator_results_{N_samples}.csv"):

        x = simulator.sample_inputs(N_samples, random_seed=42)

        # Run batch simulations with the samples generated in Cell 1
        y, x = simulator.forward_batch(x)

        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame(y)
        inputs_df = pd.DataFrame(x)

        if save:
            # Save the results to a CSV file
            results_df.to_csv(f"simulator_results_{N_samples}.csv", index=False)
            inputs_df.to_csv(f"simulator_inputs_{N_samples}.csv", index=False)

    else:
        # Read the results from the CSV file
        results_df = pd.read_csv(f"simulator_results_{N_samples}.csv")
        inputs_df = pd.read_csv(f"simulator_inputs_{N_samples}.csv")

        y = torch.tensor(results_df.to_numpy())
        x = torch.tensor(inputs_df.to_numpy())

    # These are the output summary variables we've simulated.

    # #### Train emulator with AutoEmulate
    #
    # To perform sensitivity analysis efficiently, we first need to construct an emulator—a fast, surrogate model that approximates the output of the full simulator. The simulated inputs and outputs from the cell above are  used to train the emulator, in this case we choose to use neural networks.

    # In[7]:

    ae = AutoEmulate(x, y, models=[MLP], model_tuning=False)

    # Extract the best performing emulator.

    # In[9]:

    model = ae.best_result().model

    # #### Run Sensitivity Analysis
    #
    # The emulator trained above can predict model outputs rapidly across the entire parameter space, allowing us to estimate global sensitivity measures like Sobol’ indices or Morris elementary effects without repeatedly calling the full simulator. This approach enables scalable and accurate sensitivity analysis, especially in high-dimensional or computationally intensive settings.
    #
    # Here we use AutoEmulate to perform sensitivity analysis.

    # In[10]:

    # Define the problem dictionary for Sobol sensitivity analysis
    problem = {
        "num_vars": simulator.in_dim,
        "names": simulator.param_names,
        "bounds": simulator.param_bounds,
    }

    si = SensitivityAnalysis(model, problem=problem)

    # In[11]:

    si_df = si.run(method="sobol")

    # We can select the top 5 parameters that have the biggest influcence on the pressure wave summary statistics extracted from the Nagavi Model.

    # In[14]:

    top_parameters_sa = si.top_n_sobol_params(si_df, top_n=n_parameters)
    top_parameters_sa

    # The parameters that are found to be less influential are fixed to a mid point value within its range.

    # In[15]:

    updated_range = {}
    for param_name, (min_val, max_val) in simulator.parameters_range.items():
        if param_name not in top_parameters_sa:
            print(
                f"Fixing parameter {param_name} to a value within its range ({min_val}, {max_val})"
            )
            midpoint_value = (max_val + min_val) / 2.0
            updated_range[param_name] = (midpoint_value, midpoint_value)
        else:
            updated_range[param_name] = simulator.parameters_range[
                param_name
            ]  # Fix to a value

    # In[16]:

    print("Updated parameters range with fixed values for non-sensitive parameters:")
    print(updated_range)
    simulator.parameters_range = updated_range

    # Calculate midpoint parameters
    midpoint_params_patient = []
    patient_true_values = {}
    for param_name in simulator.parameters_range:
        # Calculate the midpoint of the parameter range
        min_val, max_val = simulator.parameters_range[param_name]
        midpoint_params_patient.append((max_val + min_val) / 2.0)
        patient_true_values[param_name] = midpoint_params_patient[-1]

    # Run the simulator with midpoint parameters
    midpoint_results = simulator.forward(
        torch.tensor(midpoint_params_patient).reshape(1, -1)
    )

    # In[18]:

    # Create observations dictionary
    observations = {
        name: (val.item(), max(abs(val.item()) * 0.05, 0.05))
        for name, val in zip(simulator.output_names, midpoint_results[0])
    }
    observations

    # ### History Matching

    # In[19]:

    if not os.path.exists(f"simulator_results_{N_samples}_sa_fixed.csv"):

        x = simulator.sample_inputs(N_samples, random_seed=seed)

        # Run batch simulations with the samples generated in Cell 1
        y, x = simulator.forward_batch(x)

        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame(y)
        inputs_df = pd.DataFrame(x)

        if save:
            # Save the results to a CSV file
            results_df.to_csv(
                f"simulator_results_{N_samples}_sa_fixed.csv", index=False
            )
            inputs_df.to_csv(f"simulator_inputs_{N_samples}_sa_fixed.csv", index=False)

    else:
        # Read the results from the CSV file
        results_df = pd.read_csv(f"simulator_results_{N_samples}_sa_fixed.csv")
        inputs_df = pd.read_csv(f"simulator_inputs_{N_samples}_sa_fixed.csv")

        y = torch.tensor(results_df.to_numpy())
        x = torch.tensor(inputs_df.to_numpy())

    # only train of parameters that change
    sa_parameter_idx = [
        simulator.get_parameter_idx(param) for param in top_parameters_sa
    ]

    ae_hm = AutoEmulate(
        x[:, sa_parameter_idx],
        y,
        models=["GaussianProcess"],
        model_tuning=False,
        model_params={
            "covar_module": matern_3_2_kernel,
            "standardize_x": True,
            "standardize_y": True,
        },
    )

    res = ae_hm.best_result()
    gp_matern = res.model

    hmw = HistoryMatchingWorkflow(
        simulator=simulator,
        result=res,
        observations=observations,
        threshold=3.0,
        train_x=x.float(),
        train_y=y.float(),
        parameter_idx=sa_parameter_idx,
    )

    # Save the results
    history_matching_results = hmw.run_waves(
        n_waves=n_waves,
        n_simulations=N_samples,
        n_test_samples=2000,
        max_retries=1000,
        # only refit the emulator on the latest simulated data from the most recent wave
        refit_on_all_data=True,
        refit_emulator_on_last_wave=True,
    )

    # In[23]:

    all_df = []
    sa_parameter_idx = [
        simulator.get_parameter_idx(param) for param in top_parameters_sa
    ]

    for wave_idx, (test_parameters, impl_scores) in enumerate(history_matching_results):
        test_parameters_plausible = hmw.get_nroy(impl_scores, test_parameters)
        impl_scores_plausible = hmw.get_nroy(impl_scores, impl_scores)

        # Create DataFrame
        df = pd.DataFrame(
            test_parameters_plausible[:, sa_parameter_idx], columns=top_parameters_sa
        )
        df["Implausibility"] = impl_scores_plausible.mean(axis=1)
        df["Wave"] = wave_idx

        all_df.append(df)

    # Concatenate all waves into a single DataFrame
    result_df = pd.concat(all_df, ignore_index=True)

    # This figure shows the implausibility scores for each parameter combination, allowing us to visualize which regions of the parameter space are plausible (i.e., not ruled out) based on the observed data. The NROY region is highlighted, showing the parameters that remain after history matching.

    # In[24]:

    ref_val = [float(patient_true_values[param]) for param in top_parameters_sa]

    for wave in range(len(history_matching_results)):
        plot_implausibility(
            result_df,
            wave,
            top_parameters_sa,
            ref_val,
            output_path="implausibility_plot",
        )

    # Looking a evolution of distribution of NROY as function of the waves

    ref_val = {param: float(patient_true_values[param]) for param in top_parameters_sa}

    # Make a figure per parameter
    for i, param in enumerate(top_parameters_sa):
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=result_df, x="Wave", y=param)

        # Add horizontal line at true value
        plt.axhline(
            ref_val[param],
            color="red",
            linestyle="--",
            linewidth=2,
            label="True value" if i == 0 else None,  # add label only once
        )

        plt.title(f"Distribution of {param} by Wave")
        plt.xlabel("Wave")
        plt.ylabel(param)
        plt.tight_layout()

        # Add global legend only once (first plot)
        if i == 0:
            plt.legend(loc="upper right", frameon=True)

        plt.savefig(f"boxplot_evolution.png", dpi=300, bbox_inches="tight")

    # In[27]:

    # get the last wave results
    test_parameters, impl_scores = hmw.wave_results[-1]
    nroy_points = hmw.get_nroy(impl_scores, test_parameters)  # Implausibility < 3.0
    # Get exact min/max bounds for the parameters from the NROY points
    params_post_hm = hmw.generate_param_bounds(
        nroy_x=nroy_points, param_names=simulator.param_names, buffer_ratio=0.0
    )

    model_post_hm = hmw.emulator  # Use the emulator from history matching

    bc = BayesianCalibration(
        emulator=model_post_hm,
        parameter_range=params_post_hm,
        observations={k: torch.tensor(v[0]) for k, v in observations.items()},
        observation_noise={k: v[1] for k, v in observations.items()},
        calibration_params=top_parameters_sa,
    )

    mcmc = bc.run_mcmc(warmup_steps=250, num_samples=100, sampler="nuts")

    # In[34]:

    mcmc.summary()

    print(mcmc.get_samples())

    # We can check if the posterior samples are consistent with the true values of the parameters.

    # In[35]:

    idata = bc.to_arviz(mcmc)

    # add patient observations as a ref_val: list of floats in the order of top_parameters_sa
    # {param: float(val) for (param, val) in patient_true_values.items() if param in top_parameters_sa}
    ref_val = [float(patient_true_values[param]) for param in top_parameters_sa]

    az.plot_posterior(
        idata,
        var_names=top_parameters_sa,
        kind="hist",
        figsize=(10, 6),
        ref_val=ref_val,
    )
    plt.tight_layout()
    plt.show()
