import os
import warnings
import arviz as az
import pandas as pd
import torch

from cardiac_simulator import NaghaviSimulator
from autoemulate.calibration.bayes import BayesianCalibration
from autoemulate.calibration.history_matching import HistoryMatchingWorkflow
from autoemulate.core.compare import AutoEmulate
from autoemulate.core.sensitivity_analysis import SensitivityAnalysis
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators.gaussian_process.kernel import matern_3_2_kernel
from autoemulate.emulators.nn.mlp import MLP

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


def setup_simulator(n_samples, save=True):
    """
    Sets up the simulator, generates or loads data, and returns the simulator, inputs, and outputs.
    """
    simulator = NaghaviSimulator(
        output_variables=["lv.P"],  # Simulate left ventricle pressure
        n_cycles=300,
        dt=0.001,
    )

    if not os.path.exists(f"simulator_results_{n_samples}.csv"):
        x = simulator.sample_inputs(n_samples, random_seed=42)
        y, x = simulator.forward_batch(x)

        results_df = pd.DataFrame(y)
        inputs_df = pd.DataFrame(x)

        if save:
            results_df.to_csv(f"simulator_results_{n_samples}.csv", index=False)
            inputs_df.to_csv(f"simulator_inputs_{n_samples}.csv", index=False)
    else:
        results_df = pd.read_csv(f"simulator_results_{n_samples}.csv")
        inputs_df = pd.read_csv(f"simulator_inputs_{n_samples}.csv")
        y = torch.tensor(results_df.to_numpy())
        x = torch.tensor(inputs_df.to_numpy())

    return simulator, x, y


def train_emulator(x, y):
    """
    Trains an emulator using AutoEmulate and returns the best model.
    """
    ae = AutoEmulate(x, y, models=[MLP], model_params={})
    return ae.best_result().model


def perform_sensitivity_analysis(simulator, model, n_parameters):
    """
    Performs sensitivity analysis and returns the top parameters and updated parameter ranges.
    """
    problem = {
        "num_vars": simulator.in_dim,
        "names": simulator.param_names,
        "bounds": simulator.param_bounds,
    }
    si = SensitivityAnalysis(model, problem=problem)
    si_df = si.run(method="sobol")
    top_parameters_sa = si.top_n_sobol_params(si_df, top_n=n_parameters)

    updated_range = {}
    for param_name, (min_val, max_val) in simulator.parameters_range.items():
        if param_name not in top_parameters_sa:
            midpoint_value = (max_val + min_val) / 2.0
            updated_range[param_name] = (midpoint_value, midpoint_value)
        else:
            updated_range[param_name] = simulator.parameters_range[param_name]

    simulator.parameters_range = updated_range
    return top_parameters_sa, updated_range


def create_observations(simulator, patient_true_values):
    """
    Creates observations based on the midpoint parameters of the simulator.
    """
    midpoint_params_patient = [
        (max_val + min_val) / 2.0
        for min_val, max_val in simulator.parameters_range.values()
    ]
    midpoint_results = simulator.forward(
        torch.tensor(midpoint_params_patient).reshape(1, -1)
    )
    observations = {
        name: (val.item(), max(abs(val.item()) * 0.05, 0.05))
        for name, val in zip(simulator.output_names, midpoint_results[0])
    }
    for param_name, midpoint in zip(simulator.parameters_range.keys(), midpoint_params_patient):
        patient_true_values[param_name] = midpoint
    return observations


def run_history_matching(simulator, x, y, top_parameters_sa, observations, n_waves):
    """
    Runs history matching and returns the results and parameter bounds.
    """
    sa_parameter_idx = [simulator.get_parameter_idx(param) for param in top_parameters_sa]
    ae_hm = AutoEmulate(
        x[:, sa_parameter_idx],
        y,
        models=["GaussianProcess"],
        model_params={
            "covar_module": matern_3_2_kernel,
            "standardize_x": True,
            "standardize_y": True,
        },
    )
    res = ae_hm.best_result()
    hmw = HistoryMatchingWorkflow(
        simulator=simulator,
        result=res,
        observations=observations,
        threshold=3.0,
        train_x=x.float(),
        train_y=y.float(),
        parameter_idx=sa_parameter_idx,
    )
    history_matching_results = hmw.run_waves(
        n_waves=n_waves,
        n_simulations=len(x),
        n_test_samples=2000,
        max_retries=1000,
        refit_on_all_data=True,
        refit_emulator_on_last_wave=True,
    )
    params_post_hm = hmw.generate_param_bounds(
        nroy_x=hmw.get_nroy(*hmw.wave_results[-1]),
        param_names=simulator.param_names,
        buffer_ratio=0.0,
    )
    return history_matching_results, params_post_hm, hmw


def plot_implausibility_results(history_matching_results, hmw, top_parameters_sa, patient_true_values):
    """
    Plots implausibility results for each wave.
    """
    ref_val = [float(patient_true_values[param]) for param in top_parameters_sa]
    all_df = []
    for wave_idx, (test_parameters, impl_scores) in enumerate(history_matching_results):
        test_parameters_plausible = hmw.get_nroy(impl_scores, test_parameters)
        impl_scores_plausible = hmw.get_nroy(impl_scores, impl_scores)
        df = pd.DataFrame(
            test_parameters_plausible[:, [simulator.get_parameter_idx(param) for param in top_parameters_sa]],
            columns=top_parameters_sa,
        )
        df["Implausibility"] = impl_scores_plausible.mean(axis=1)
        df["Wave"] = wave_idx
        all_df.append(df)
    result_df = pd.concat(all_df, ignore_index=True)

    for wave in range(len(history_matching_results)):
        plot_implausibility(result_df, wave, top_parameters_sa, ref_val, output_path="implausibility_plot")


def run_bayesian_calibration(model_post_hm, params_post_hm, observations, top_parameters_sa):
    """
    Runs Bayesian calibration and returns the MCMC results.
    """
    bc = BayesianCalibration(
        emulator=model_post_hm,
        parameter_range=params_post_hm,
        observations={k: torch.tensor(v[0]) for k, v in observations.items()},
        observation_noise={k: v[1] for k, v in observations.items()},
        calibration_params=top_parameters_sa,
    )
    return bc.run_mcmc(warmup_steps=250, num_samples=100, sampler="nuts")


def main():
    seed = 42
    set_random_seed(seed)

    n_samples = 1024
    n_waves = 5
    n_parameters = 2

    simulator, x, y = setup_simulator(n_samples)
    model = train_emulator(x, y)
    top_parameters_sa, updated_range = perform_sensitivity_analysis(simulator, model, n_parameters)

    patient_true_values = {}
    observations = create_observations(simulator, patient_true_values)

    history_matching_results, params_post_hm, hmw = run_history_matching(
        simulator, x, y, top_parameters_sa, observations, n_waves
    )
    plot_implausibility_results(history_matching_results, hmw, top_parameters_sa, patient_true_values)

    mcmc = run_bayesian_calibration(hmw.emulator, params_post_hm, observations, top_parameters_sa)
    print(mcmc.summary())


if __name__ == "__main__":
    main()