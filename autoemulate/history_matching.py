import numpy as np


def history_matching(obs, predictions, threshold=3.0, discrepancy=0.0, rank=1):
    """
    Perform history matching to compute implausibility and identify NROY and RO points.
    This implementation performs history matching as a single run, completing the process
    in one execution without iterative refinement or staged waves.

    The implausibility is calculated as the absolute difference between the observed and
    predicted values, normalized by the square root of the sum of the variances of the
    observed and predicted values. The implausibility is then compared to a threshold to
    classify the points as NROY or RO. The discrepancy value(s) can be provided as a
    scalar or an array to account for model discrepancy.
    The rank parameter is used to select the number of observations to consider for implausibility calculation.
    The default value is 1, which corresponds to the most recent observation.
    Parameters:
        obs (tuple): Observations as (mean, variance).
        predictions (tuple): Predicted (mean, variance).
        threshold (float): Implausibility threshold for NROY classification.
        discrepancy (float or ndarray): Discrepancy value(s).
        rank (int): Rank for implausibility calculation.

    Returns:
        dict: Contains implausibility (I), NROY indices, and RO indices.
    """
    obs_mean, obs_var = np.atleast_1d(obs[0]), np.atleast_1d(obs[1])
    pred_mean, pred_var = np.atleast_1d(predictions[0]), np.atleast_1d(predictions[1])
    if len(obs_mean) != len(pred_mean[1]):
        raise ValueError(
            "The number of means in observations and predictions must be equal."
        )
    if len(obs_var) != len(pred_var[1]):
        raise ValueError(
            "The number of variances in observations and predictions must be equal."
        )
    discrepancy = np.atleast_1d(discrepancy)
    n_obs = len(obs_mean)
    rank = min(max(rank, 0), n_obs - 1)
    if discrepancy.size == 1:
        discrepancy = np.full(n_obs, discrepancy)

    Vs = pred_var + discrepancy + obs_var
    I = np.abs(obs_mean - pred_mean) / np.sqrt(Vs)

    NROY = np.where(I <= threshold)[0]
    RO = np.where(I > threshold)[0]

    return {"I": I, "NROY": list(NROY), "RO": list(RO)}
