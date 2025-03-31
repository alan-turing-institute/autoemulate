import numpy as np


class HistoryMatching:
    def __init__(self, threshold=3.0, discrepancy=0.0, rank=1):
        self.threshold = threshold
        self.discrepancy = discrepancy
        self.rank = rank

    def history_matching(self, obs, predictions):
        """
        Perform history matching to compute implausibility and identify NROY and RO points.
        """
        obs_mean, obs_var = np.atleast_1d(obs[0]), np.atleast_1d(obs[1])
        pred_mean, pred_var = np.atleast_1d(predictions[0]), np.atleast_1d(predictions[1])
        if len(obs_mean) != len(pred_mean):
            raise ValueError("The number of means in observations and predictions must be equal.")
        if len(obs_var) != len(pred_var):
            raise ValueError("The number of variances in observations and predictions must be equal.")
        
        discrepancy = np.atleast_1d(self.discrepancy)
        n_obs = len(obs_mean)
        rank = min(max(self.rank, 0), n_obs - 1)
        if discrepancy.size == 1:
            discrepancy = np.full(n_obs, discrepancy)
        
        Vs = pred_var + discrepancy + obs_var
        I = np.abs(obs_mean - pred_mean) / np.sqrt(Vs)
        
        NROY = np.where(I <= self.threshold)[0]
        RO = np.where(I > self.threshold)[0]
        
        return {"I": I, "NROY": list(NROY), "RO": list(RO)}
    
    def _sample_new_points(self, X_nroy, n_points):
        """
        Sample new points uniformly within the NROY region.
        """
        min_bounds = np.min(X_nroy, axis=0)
        max_bounds = np.max(X_nroy, axis=0)
        return np.random.uniform(min_bounds, max_bounds, size=(n_points, X_nroy.shape[1]))
