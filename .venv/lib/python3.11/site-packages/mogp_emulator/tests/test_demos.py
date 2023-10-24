import pytest
import runpy
import pathlib
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

scripts = [
    pathlib.Path(__file__, "../..", "demos/gp_demos.py"),
    pathlib.Path(__file__, "../..", "demos/gp_kernel_demos.py"),
    pathlib.Path(__file__, "../..", "demos/historymatch_demos.py"),
    pathlib.Path(__file__, "../..", "demos/kdr_demos.py"),
    pathlib.Path(__file__, "../..", "demos/mice_demos.py"),
    pathlib.Path(__file__, '../..', 'demos/multioutput_tutorial.py'),
    pathlib.Path(__file__, '../..', 'demos/tutorial.py'),
    pathlib.Path(__file__, '../..', 'demos/excalibur_workshop_demo.py'),
]

@pytest.mark.skip
@pytest.mark.parametrize("script", scripts)
def test_script_execution(script):
    RandomState(MT19937(SeedSequence(987654321)))
    runpy.run_path(script)
    if makeplots:
        plt.close()
