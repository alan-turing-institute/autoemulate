import torch

from autoemulate.experimental.simulations.base import Simulator
from autoemulate.experimental.types import TensorLike
from autoemulate.simulations.projectile import (
    simulate_projectile,
    simulate_projectile_multioutput,
)


class Projectile(Simulator):
    """
    Simulator of projectile motion.
    """

    def __init__(
        self,
        param_ranges=None,
        output_names=None,
    ):
        if param_ranges is None:
            param_ranges = {"c": (-5.0, 1.0), "v0": (0.0, 1000)}
        if output_names is None:
            output_names = ["distance"]
        super().__init__(param_ranges, output_names)

    def _forward(self, x: TensorLike) -> TensorLike:
        """
        Parameters
        ----------
        x : TensorLike
            Dictionary of input parameter values to simulate:
            - `c`: the drag coefficient on a log scale
            - `v0`: velocity

        Returns
        -------
        TensorLike
            Distance travelled by projectile.
        """
        y = simulate_projectile(x.numpy()[0])
        return torch.tensor([y]).view(-1, 1)


class ProjectileMultioutput(Simulator):
    """
    Simulator of projectile motion.
    """

    def __init__(
        self,
        param_ranges=None,
        output_names=None,
    ):
        if param_ranges is None:
            param_ranges = {"c": (-5.0, 1.0), "v0": (0.0, 1000)}
        if output_names is None:
            output_names = ["distance", "impact_velocity"]
        super().__init__(param_ranges, output_names)

    def _forward(self, x: TensorLike) -> TensorLike:
        """
        Parameters
        ----------
        x : TensorLike
            Dictionary of input parameter values to simulate:
            - `c`: the drag coefficient on a log scale
            - `v0`: velocity

        Returns
        -------
        TensorLike
            Distance travelled by projectile and impact velocity.
        """
        y = simulate_projectile_multioutput(x.numpy()[0])
        return torch.tensor([y])
