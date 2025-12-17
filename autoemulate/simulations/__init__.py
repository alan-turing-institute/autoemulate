from .epidemic import Epidemic
from .flow_problem import FlowProblem
from .projectile import Projectile, ProjectileMultioutput
from .seir import SEIRSimulator

ALL_SIMULATORS = [
    Epidemic,
    SEIRSimulator,
    FlowProblem,
    Projectile,
    ProjectileMultioutput,
]

__all__ = [
    "Epidemic",
    "FlowProblem",
    "Projectile",
    "ProjectileMultioutput",
    "SEIRSimulator",
]

SIMULATOR_REGISTRY = dict(zip(__all__, ALL_SIMULATORS, strict=False))
