# autoemulate

![CI](https://github.com/alan-turing-institute/autoemulate/actions/workflows/ci.yaml/badge.svg)
[![codecov](https://codecov.io/gh/alan-turing-institute/autoemulate/graph/badge.svg?token=XD1HXQUIGK)](https://codecov.io/gh/alan-turing-institute/autoemulate)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


<!-- SPHINX-START -->
Simulations of physical systems are often slow and need lots of compute, which makes them unpractical for applications like digital twins. The goal of `autoemulate` is to make it easy to replace simulations with fast, accurate emulator models.

The project is in very early development. 


# usage

```python
import numpy as np
from autoemulate.compare import AutoEmulate
from autoemulate.experimental_design import LatinHypercube
from autoemulate.demos.projectile import simulator

# sample from a simulation
lhd = LatinHypercube([(-5., 1.), (0., 1000.)])
X = lhd.sample(100)
y = np.array([simulator(x) for x in X])

# compare emulators and select best
ae = AutoEmulate()
ae.setup(X, y)
ae.compare() 

# evaluate
ae.print_results()

# save & load
ae.save_model("best_model")
best_emulator = ae.load_model("best_model")
```