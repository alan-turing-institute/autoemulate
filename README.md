# AutoEmulate

![CI](https://github.com/alan-turing-institute/autoemulate/actions/workflows/ci.yaml/badge.svg)
[![codecov](https://codecov.io/gh/alan-turing-institute/autoemulate/graph/badge.svg?token=XD1HXQUIGK)](https://codecov.io/gh/alan-turing-institute/autoemulate)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![All Contributors](https://img.shields.io/github/all-contributors/alan-turing-institute/autoemulate?color=ee8449&style=flat-square)](#contributors)
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://alan-turing-institute.github.io/autoemulate/)

<!-- SPHINX-START -->

Simulations of physical systems are often slow and need lots of compute, which makes them unpractical for applications like digital twins, or when they have to run thousands of times to do uncertainty quantification or sensitivity analyses. The goal of `AutoEmulate` is to make it easy to replace simulations with fast, accurate emulators. To do this, `AutoEmulate` automatically fits and compares lots of models, like *Radial Basis Functions*, *Gaussian Processes* or *Neural Networks* to find the best emulator for a simulation.

The project is in early development. 

<img src="https://raw.githubusercontent.com/alan-turing-institute/autoemulate/main/misc/robot2.png" alt="" width="38.2%">

## installation

```bash
pip install autoemulate
```

development version from GitHub:

```bash
pip install git+https://github.com/alan-turing-institute/autoemulate.git
```

or for contributors using [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/alan-turing-institute/autoemulate.git
cd autoemulate
poetry install
```

## quick start

```python
import numpy as np
from autoemulate.compare import AutoEmulate
from autoemulate.experimental_design import LatinHypercube
from autoemulate.simulations.projectile import simulate_projectile

# sample from a simulation
lhd = LatinHypercube([(-5., 1.), (0., 1000.)])
X = lhd.sample(100)
y = np.array([simulate_projectile(x) for x in X])
# compare emulator models
ae = AutoEmulate()
ae.setup(X, y)
best_model = ae.compare() 
# training set cross-validation results
ae.print_results() 
# test set results for the best model
ae.evaluate_model(best_model) 
# refit on full data
best_emulator = ae.refit_model(best_model) 
# emulate
best_emulator.predict(X)
```

## documentation

You can find tutorials, FAQs and the API reference [here](https://alan-turing-institute.github.io/autoemulate/). The documentation is still work in progress.

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://www.westerling.nu"><img src="https://avatars.githubusercontent.com/u/7298727?v=4?s=100" width="100px;" alt="Kalle Westerling"/><br /><sub><b>Kalle Westerling</b></sub></a><br /><a href="#doc-kallewesterling" title="Documentation">📖</a> <a href="#code-kallewesterling" title="Code">💻</a> <a href="#content-kallewesterling" title="Content">🖋</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://bryanli.io"><img src="https://avatars.githubusercontent.com/u/9648242?v=4?s=100" width="100px;" alt="Bryan M. Li"/><br /><sub><b>Bryan M. Li</b></sub></a><br /><a href="#code-bryanlimy" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mastoffel"><img src="https://avatars.githubusercontent.com/u/7348440?v=4?s=100" width="100px;" alt="martin"/><br /><sub><b>martin</b></sub></a><br /><a href="#code-mastoffel" title="Code">💻</a> <a href="#ideas-mastoffel" title="Ideas, Planning, & Feedback">🤔</a> <a href="#doc-mastoffel" title="Documentation">📖</a> <a href="#maintenance-mastoffel" title="Maintenance">🚧</a> <a href="#research-mastoffel" title="Research">🔬</a> <a href="#review-mastoffel" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/edaub"><img src="https://avatars.githubusercontent.com/u/45598892?v=4?s=100" width="100px;" alt="Eric Daub"/><br /><sub><b>Eric Daub</b></sub></a><br /><a href="#ideas-edaub" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-edaub" title="Project Management">📆</a> <a href="#review-edaub" title="Reviewed Pull Requests">👀</a> <a href="#code-edaub" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/snie007"><img src="https://avatars.githubusercontent.com/u/20723650?v=4?s=100" width="100px;" alt="steven niederer"/><br /><sub><b>steven niederer</b></sub></a><br /><a href="#ideas-snie007" title="Ideas, Planning, & Feedback">🤔</a> <a href="#content-snie007" title="Content">🖋</a> <a href="#projectManagement-snie007" title="Project Management">📆</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MaxBalmus"><img src="https://avatars.githubusercontent.com/u/34339336?v=4?s=100" width="100px;" alt="Maximilian Balmus"/><br /><sub><b>Maximilian Balmus</b></sub></a><br /><a href="#code-MaxBalmus" title="Code">💻</a> <a href="#bug-MaxBalmus" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
