import numpy as np


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
    if discrepancy.size == 1:
        discrepancy = np.full(n_obs, discrepancy)

    Vs = pred_var + discrepancy + obs_var
    I = np.abs(obs_mean - pred_mean) / np.sqrt(Vs)

    NROY = np.where(I <= threshold)[0]
    RO = np.where(I > threshold)[0]

    return {"I": I, "NROY": list(NROY), "RO": list(RO)}
