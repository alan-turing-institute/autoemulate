from .epidemic import Epidemic
from .projectile import Projectile, ProjectileMultioutput

ALL_SIMULATORS = [Epidemic, Projectile, ProjectileMultioutput]

__all__ = ["Epidemic", "Projectile", "ProjectileMultioutput"]

SIMULATOR_FROM_STR = dict(zip(__all__, ALL_SIMULATORS, strict=False))
