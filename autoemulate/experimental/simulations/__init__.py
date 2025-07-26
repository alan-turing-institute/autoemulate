from .epidemic import Epidemic
from .flow_problem import FlowProblem
from .projectile import Projectile, ProjectileMultioutput

ALL_SIMULATORS = [Epidemic, FlowProblem, Projectile, ProjectileMultioutput]

__all__ = ["Epidemic", "FlowProblem", "Projectile", "ProjectileMultioutput"]

SIMULATOR_REGISTRY = dict(zip(__all__, ALL_SIMULATORS, strict=False))
