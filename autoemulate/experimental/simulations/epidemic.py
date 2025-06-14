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
        param_ranges={"beta": (0.1, 0.5), "gamma": (0.01, 0.2)},
        output_names=["infection_rate"],
    ):
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
        y = simulate_epidemic(x.numpy()[0])
        return torch.tensor([y]).view(-1, 1)
