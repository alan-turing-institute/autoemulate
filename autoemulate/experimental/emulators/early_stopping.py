from copy import deepcopy

import torch
from torch import nn


class EarlyStopping:
    """
    Stop training early if the training loss did not improve in `patience` number of
    epochs by at least `threshold` value. Can work with any PyTorch model.

    Parameters
    ----------
    lower_is_better: bool
        Whether lower scores should be considered an improvement. Defaults to True.
    patience: int
        Number of epochs to wait for improvement of the monitored value until the
        training process is stopped. Defaults to 5.
    threshold: int
        Minimum change in monitored score value (i.e., `min_delta`) that is considered
        an improvement in perforamnce. Defaults to 1e-4.
    threshold_mode: str
        One of `rel`, `abs`. Decides whether the `threshold` value is interpreted in
        absolute terms or as a fraction of the best  score so far (relative). Defaults
        to `rel`.
    load_best: bool
        Whether to restore module weights from the epoch with the best value of the
        monitored quantity. If False, the module weights obtained at the last step of
        training are used. Defaults to False.

    Notes
    -----
    This class is almost identical to `EarlyStopping` in skorch. The main difference
    is that the method `_calc_new_threshold`, is corrected to ensure monotonicity.
    We also do not have the option to monitor validation loss instead of train loss.
    """

    def __init__(
        self,
        patience: int = 5,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        lower_is_better: bool = True,
        load_best: bool = False,
    ):
        self.lower_is_better = lower_is_better
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.misses_ = 0
        self.dynamic_threshold_ = None
        self.load_best = load_best

    def __getstate__(self):
        # Avoids to save the module_ weights twice when pickling
        state = self.__dict__.copy()
        state["best_model_weights_"] = None
        return state

    def on_train_begin(self):
        if self.threshold_mode not in ["rel", "abs"]:
            raise ValueError(f"Invalid threshold mode: '{self.threshold_mode}'")
        self.misses_ = 0
        self.dynamic_threshold_ = torch.inf if self.lower_is_better else -torch.inf
        self.best_model_weights_ = None
        self.best_epoch_ = 0

    def on_epoch_end(self, model: nn.Module, curr_epoch: int, curr_score: float):
        """
        Determine whether to stop training at this epoch and save best model weights.

        Parameters
        ----------
        model: nn.Module
            A PyTorch model.
        curr_epoch: int
            The current epoch.
        curr_score: float
            The current training loss.
        """

        if not self._is_score_improved(curr_score):
            self.misses_ += 1
        else:
            self.misses_ = 0
            self.dynamic_threshold_ = self._calc_new_threshold(curr_score)
            self.best_epoch_ = curr_epoch
            if self.load_best:
                self.best_model_weights_ = deepcopy(model.state_dict())
        if self.misses_ == self.patience:
            print(
                "Stopping since train loss has not improved in the last "
                f"{self.patience} epochs."
            )
            raise KeyboardInterrupt

    def on_train_end(self, model: nn.Module, last_epoch: int):
        """
        Optionally restore module weights from the epoch with the best train loss.

        Parameters
        ----------
        model: nn.Module
            A PyTorch model.
        last_epoch: int
            The number of completed epochs before training was stopped.
        """
        if (
            self.load_best
            and (self.best_epoch_ != last_epoch)
            and (self.best_model_weights_ is not None)
        ):
            model.load_state_dict(self.best_model_weights_)

    def _is_score_improved(self, score):
        if self.lower_is_better:
            return score < self.dynamic_threshold_
        return score > self.dynamic_threshold_

    def _calc_new_threshold(self, score):
        """
        Determine threshold based on score.

        Notes
        -----
        This function updates the skorch version, which assumes that the score returned
        by the cost function is always positive.
        PR to skorch pending: https://github.com/skorch-dev/skorch/pull/1065
        """
        if self.threshold_mode == "rel":
            abs_threshold_change = self.threshold * abs(score)
        else:
            abs_threshold_change = self.threshold

        if self.lower_is_better:
            new_threshold = score - abs_threshold_change
        else:
            new_threshold = score + abs_threshold_change
        return new_threshold
