import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error


def rmse(y_true, y_pred, multioutput="uniform_average"):
    """Returns the root mean squared error.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_outputs)
        Simulation output.
    y_pred : array-like, shape (n_samples, n_outputs)
        Emulator output.
    multioutput : str, {"raw_values", "uniform_average", "variance_weighted"}, default="uniform_average"
        Defines how to aggregate metric for each output.
    """
    return root_mean_squared_error(y_true, y_pred, multioutput=multioutput)


def r2(y_true, y_pred, multioutput="uniform_average"):
    """Returns the R^2 score.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_outputs)
        Simulation output.
    y_pred : array-like, shape (n_samples, n_outputs)
        Emulator output.
    multioutput : str, {"raw_values", "uniform_average", "variance_weighted"}, default="uniform_average"
        Defines how to aggregate metric for each output.
    """
    return r2_score(y_true, y_pred, multioutput=multioutput)


#: A dictionary of available metrics.
METRIC_REGISTRY = {
    "rmse": rmse,
    "r2": r2,
}


def history_matching(obs, expectations, threshold=3.0, discrepancy=0.0, rank=1):
    """
    Perform history matching to compute implausibility and identify NROY and RO points.

    Parameters:
        obs (tuple): Observations as (mean, variance).
        expectations (tuple): Predicted (mean, variance).
        threshold (float): Implausibility threshold for NROY classification.
        discrepancy (float or ndarray): Discrepancy value(s).
        rank (int): Rank for implausibility calculation.

    Returns:
        dict: Contains implausibility (I), NROY indices, and RO indices.
    """
    obs_mean, obs_var = np.atleast_1d(obs[0]), np.atleast_1d(obs[1])
    pred_mean, pred_var = np.atleast_1d(expectations[0]), np.atleast_1d(expectations[1])

    discrepancy = np.atleast_1d(discrepancy)
    n_obs = len(obs_mean)
    rank = min(max(rank, 0), n_obs - 1)
    #  Vs represents the total variance associated with the observations, predictions, and potential discrepancies.
    Vs = pred_var + discrepancy[:, np.newaxis] + obs_var[:, np.newaxis]
    I = np.abs(obs_mean[:, np.newaxis] - pred_mean) / np.sqrt(Vs)
    I_ranked = np.partition(I, rank, axis=0)[rank]

    NROY = np.where(I_ranked <= threshold)[0]
    RO = np.where(I_ranked > threshold)[0]

    return {"I": I_ranked, "NROY": list(NROY), "RO": list(RO)}


def negative_log_likelihood(model_params, obs_mean, obs_var):
    """
    Compute the negative log-likelihood.

    Parameters:
        model_params (Tensor): Model parameters [mean, variance] as a PyTorch tensor.
        obs_mean (Tensor): Observed mean (float or tensor).
        obs_var (Tensor): Observed variance (float or tensor).

    Returns:
        Tensor: Negative log-likelihood value.
    """
    model_mean, model_var = model_params
    log_likelihood = 0.5 * torch.log(2 * torch.pi * obs_var) + (
        obs_mean - model_mean
    ) ** 2 / (2 * obs_var)
    return log_likelihood.sum()  # Sum over all observations


def max_likelihood(expectations, obs, lr=0.01, epochs=1000):
    """
    Perform Maximum Likelihood Estimation (MLE) using PyTorch to optimize parameters.

    Parameters:
        obs (tuple): Observations as (mean, variance).
        expectations (tuple): Predicted (mean, variance).
        lr (float): Learning rate for optimizer.
        epochs (int): Number of optimization epochs.

    Returns:
        dict: Contains the log-likelihoods and plausible region indices.
    """
    # Convert observed values to PyTorch tensors
    obs_mean = torch.tensor(obs[0], dtype=torch.float32)
    obs_var = torch.tensor(obs[1], dtype=torch.float32)

    # Convert GP predictions to PyTorch tensors
    gp_means = torch.tensor(expectations[0], dtype=torch.float32)
    gp_vars = torch.tensor(expectations[1], dtype=torch.float32)

    # Track negative log-likelihoods for each parameter set
    NLL = []

    for mean, var in zip(gp_means, gp_vars):
        params = torch.tensor([mean.item(), var.item()], requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=lr)

        # Optimize parameters
        for _ in range(epochs):
            optimizer.zero_grad()
            neg_log_likelihood = negative_log_likelihood(params, obs_mean, obs_var)
            neg_log_likelihood.backward()  # Compute gradients
            optimizer.step()

        NLL.append(negative_log_likelihood(params, obs_mean, obs_var).item())

    NLLs = torch.tensor(NLL)

    # Define plausible regions: top 5% of likelihoods
    threshold = torch.quantile(NLLs, 0.20)
    plausible_indices = torch.where(NLLs <= threshold)[0]

    return {
        "LLs": NLLs.numpy(),
        "plausible_indices": plausible_indices.numpy(),
        "plausible_LLs": NLLs[plausible_indices].numpy(),
    }
