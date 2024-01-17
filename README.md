# autoemulate

![CI](https://github.com/alan-turing-institute/autoemulate/actions/workflows/ci.yaml/badge.svg)
[![codecov](https://codecov.io/gh/alan-turing-institute/autoemulate/graph/badge.svg?token=XD1HXQUIGK)](https://codecov.io/gh/alan-turing-institute/autoemulate)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


<!-- SPHINX-START -->

Simulations of physical systems are often slow and need lots of compute, which makes them unpractical for applications like digital twins, or in situations where they have to run thousands of times, like sensitivity analyses. The goal of `autoemulate` is to make it easy to replace simulations with fast, accurate emulators. To do this, `autoemulate` automatically fits and compares lots of models, like *Radial Basis Functions*, *Gaussian Processes* or *Neural Networks* to find the best emulator for a simulation.

The project is in very early development. 

<img src="misc/robot2.png" alt="emulating simulations with ML" width="61.8%">

## setup 

using [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/alan-turing-institute/autoemulate.git
cd autoemulate
poetry install
poetry shell
```

## quick start

```python
import numpy as np
from autoemulate.compare import AutoEmulate
from autoemulate.experimental_design import LatinHypercube
from autoemulate.demos.projectile import simulator

# sample from a simulation
lhd = LatinHypercube([(-5., 1.), (0., 1000.)])
X = lhd.sample(100)
y = np.array([simulator(x) for x in X])

# compare emulator models
ae = AutoEmulate()
ae.setup(X, y)
ae.compare() 

# evaluate
ae.print_results()

# save & load best model
ae.save_model("best_model")
best_emulator = ae.load_model("best_model")

# emulate
best_emulator.predict(X)
```

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://www.westerling.nu"><img src="https://avatars.githubusercontent.com/u/7298727?v=4?s=100" width="100px;" alt="Kalle Westerling"/><br /><sub><b>Kalle Westerling</b></sub></a><br /><a href="#doc-kallewesterling" title="Documentation">📖</a> <a href="#code-kallewesterling" title="Code">💻</a> <a href="#content-kallewesterling" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://bryanli.io"><img src="https://avatars.githubusercontent.com/u/9648242?v=4?s=100" width="100px;" alt="Bryan M. Li"/><br /><sub><b>Bryan M. Li</b></sub></a><br /><a href="#code-bryanlimy" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->