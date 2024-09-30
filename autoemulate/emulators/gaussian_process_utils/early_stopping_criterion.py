import numpy as np
from skorch.callbacks import EarlyStopping


class EarlyStoppingCustom(EarlyStopping):
    """Callback for stopping training when scores don't improve.

    Stop training early if a specified `monitor` metric did not
    improve in `patience` number of epochs by at least `threshold`.
    **Note**: This version is virtually identical to `EarlyStopping`,
    with the difference being that the method `_calc_new_threshold`,
    is corrected to ensure monotonicity.

    also see https://github.com/skorch-dev/skorch/pull/1065

    Parameters
    ----------
    monitor : str (default='valid_loss')
      Value of the history to monitor to decide whether to stop
      training or not.  The value is expected to be double and is
      commonly provided by scoring callbacks such as
      :class:`skorch.callbacks.EpochScoring`.

    lower_is_better : bool (default=True)
      Whether lower scores should be considered better or worse.

    patience : int (default=5)
      Number of epochs to wait for improvement of the monitor value
      until the training process is stopped.

    threshold : int (default=1e-4)
      Ignore score improvements smaller than `threshold`.

    threshold_mode : str (default='rel')
        One of `rel`, `abs`. Decides whether the `threshold` value is
        interpreted in absolute terms or as a fraction of the best
        score so far (relative)

    sink : callable (default=print)
      The target that the information about early stopping is
      sent to. By default, the output is printed to stdout, but the
      sink could also be a logger or :func:`~skorch.utils.noop`.

    load_best: bool (default=False)
      Whether to restore module weights from the epoch with the best value of
      the monitored quantity. If False, the module weights obtained at the
      last step of training are used. Note that only the module is restored.
      Use the ``Checkpoint`` callback with the :attr:`~Checkpoint.load_best`
      argument set to ``True`` if you need to restore the whole object.

    """

    def _calc_new_threshold(self, score):
        """Determine threshold based on score."""
        if self.threshold_mode == "rel":
            abs_threshold_change = self.threshold * np.abs(score)
        else:
            abs_threshold_change = self.threshold

        if self.lower_is_better:
            new_threshold = score - abs_threshold_change
        else:
            new_threshold = score + abs_threshold_change
        return new_threshold
