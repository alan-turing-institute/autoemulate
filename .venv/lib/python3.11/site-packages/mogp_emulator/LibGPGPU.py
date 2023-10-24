"""
Wrapper for the optional GPU interface library, libgpgpu.so
"""

HAVE_LIBGPGPU = False
try:
    from libgpgpu import *
    HAVE_LIBGPGPU = True
except ModuleNotFoundError:
    pass


def gpu_usable():
    return HAVE_LIBGPGPU and have_compatible_device()
