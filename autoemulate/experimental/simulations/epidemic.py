import torch

from autoemulate.experimental.simulations.base import Simulator
from autoemulate.experimental.types import TensorLike
from autoemulate.simulations.epidemic import simulate_epidemic


class Epidemic(Simulator):
    """
    Simulator of infectious disease spread (SIR).
    """

    def __init__(
        self,
        param_ranges=None,
        output_names=None,
    ):
        if param_ranges is None:
            param_ranges = {"beta": (0.1, 0.5), "gamma": (0.01, 0.2)}
        if output_names is None:
            output_names = ["infection_rate"]
        super().__init__(param_ranges, output_names)

    def _forward(self, x: TensorLike) -> TensorLike:
        """
        Parameters
        ----------
        x : TensorLike
            input parameter values to simulate [beta, gamma]:
            - `beta`: the transimission rate per day
            - `gamma`: the recovery rate per day

        Returns
        -------
        TensorLike
            Peak infection rate.
        """
        y = simulate_epidemic(x.cpu().numpy()[0])
        # TODO (#537): update with default dtype
        return torch.tensor([y], dtype=torch.float32).view(-1, 1)
