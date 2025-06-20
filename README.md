# AutoEmulate <a href="https://alan-turing-institute.github.io/autoemulate/"><img src="misc/AE_logo_final.png" align="right" height="138" /></a>

![CI](https://github.com/alan-turing-institute/autoemulate/actions/workflows/ci.yaml/badge.svg)
[![codecov](https://codecov.io/gh/alan-turing-institute/autoemulate/graph/badge.svg?token=XD1HXQUIGK)](https://codecov.io/gh/alan-turing-institute/autoemulate)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![All Contributors](https://img.shields.io/github/all-contributors/alan-turing-institute/autoemulate?color=ee8449&style=flat-square)](#contributors)
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://alan-turing-institute.github.io/autoemulate/)

<!-- SPHINX-START -->
Simulations of physical systems are often slow and need lots of compute, which makes them unpractical for real-world applications like digital twins, or when they have to run thousands of times for sensitivity analyses. The goal of `AutoEmulate` is to make it easy to replace simulations with fast, accurate emulators. To do this, `AutoEmulate` automatically fits and compares various emulators, ranging from simple models like Radial Basis Functions and Second Order Polynomials to more complex models like Support Vector Machines and  Gaussian Processes to find the best emulator for a simulation. 

⚠️ Warning: This is an early version of the package and is still under development. We are working on improving the documentation and adding more features. If you have any questions or suggestions, please open an issue or a pull request.

## Documentation

You can find the project documentation [here](https://alan-turing-institute.github.io/autoemulate/), including [installation](https://alan-turing-institute.github.io/autoemulate/getting-started/installation.html).

## The AutoEmulate project

- The AutoEmulate project is run out of the [Alan Turing Institute](https://www.turing.ac.uk/).
- Visit [autoemulate.com](https://www.autoemulate.com/) to learn more.
- We have also published a paper in [The Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.07626). 

  Please cite this paper if you use the package in your work:

  ```bibtex
  @article{Stoffel2025, doi = {10.21105/joss.07626}, url = {https://doi.org/10.21105/joss.07626}, year = {2025}, publisher = {The Open Journal}, volume = {10}, number = {107}, pages = {7626}, author = {Martin A. Stoffel and Bryan M. Li and Kalle Westerling and Sophie Arana and Max Balmus and Eric Daub and Steve Niederer}, title = {AutoEmulate: A Python package for semi-automated emulation}, journal = {Journal of Open Source Software} }
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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mastoffel"><img src="https://avatars.githubusercontent.com/u/7348440?v=4?s=100" width="100px;" alt="martin"/><br /><sub><b>martin</b></sub></a><br /><a href="#code-mastoffel" title="Code">💻</a> <a href="#ideas-mastoffel" title="Ideas, Planning, & Feedback">🤔</a> <a href="#doc-mastoffel" title="Documentation">📖</a> <a href="#maintenance-mastoffel" title="Maintenance">🚧</a> <a href="#research-mastoffel" title="Research">🔬</a> <a href="#review-mastoffel" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/edaub"><img src="https://avatars.githubusercontent.com/u/45598892?v=4?s=100" width="100px;" alt="Eric Daub"/><br /><sub><b>Eric Daub</b></sub></a><br /><a href="#ideas-edaub" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-edaub" title="Project Management">📆</a> <a href="#review-edaub" title="Reviewed Pull Requests">👀</a> <a href="#code-edaub" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/snie007"><img src="https://avatars.githubusercontent.com/u/20723650?v=4?s=100" width="100px;" alt="steven niederer"/><br /><sub><b>steven niederer</b></sub></a><br /><a href="#ideas-snie007" title="Ideas, Planning, & Feedback">🤔</a> <a href="#content-snie007" title="Content">🖋</a> <a href="#projectManagement-snie007" title="Project Management">📆</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MaxBalmus"><img src="https://avatars.githubusercontent.com/u/34339336?v=4?s=100" width="100px;" alt="Maximilian Balmus"/><br /><sub><b>Maximilian Balmus</b></sub></a><br /><a href="#code-MaxBalmus" title="Code">💻</a> <a href="#bug-MaxBalmus" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aranas"><img src="https://avatars.githubusercontent.com/u/6906140?v=4?s=100" width="100px;" alt="Sophie Arana"/><br /><sub><b>Sophie Arana</b></sub></a><br /><a href="#content-aranas" title="Content">🖋</a> <a href="#doc-aranas" title="Documentation">📖</a> <a href="#projectManagement-aranas" title="Project Management">📆</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aduncan001"><img src="https://avatars.githubusercontent.com/u/2352812?v=4?s=100" width="100px;" alt="Andrew Duncan"/><br /><sub><b>Andrew Duncan</b></sub></a><br /><a href="#ideas-aduncan001" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-aduncan001" title="Project Management">📆</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/marjanfamili"><img src="https://avatars.githubusercontent.com/u/44607686?v=4?s=100" width="100px;" alt="Marjan Famili"/><br /><sub><b>Marjan Famili</b></sub></a><br /><a href="#code-marjanfamili" title="Code">💻</a> <a href="#ideas-marjanfamili" title="Ideas, Planning, & Feedback">🤔</a> <a href="#doc-marjanfamili" title="Documentation">📖</a> <a href="#review-marjanfamili" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/radka-j"><img src="https://avatars.githubusercontent.com/u/29207091?v=4?s=100" width="100px;" alt="Radka Jersakova"/><br /><sub><b>Radka Jersakova</b></sub></a><br /><a href="#code-radka-j" title="Code">💻</a> <a href="#projectManagement-radka-j" title="Project Management">📆</a> <a href="#maintenance-radka-j" title="Maintenance">🚧</a> <a href="#ideas-radka-j" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://cisprague.github.io/"><img src="https://avatars.githubusercontent.com/u/17131395?v=4?s=100" width="100px;" alt="Christopher Iliffe Sprague"/><br /><sub><b>Christopher Iliffe Sprague</b></sub></a><br /><a href="#code-cisprague" title="Code">💻</a> <a href="#design-cisprague" title="Design">🎨</a> <a href="#ideas-cisprague" title="Ideas, Planning, & Feedback">🤔</a> <a href="#review-cisprague" title="Reviewed Pull Requests">👀</a> <a href="#doc-cisprague" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.energy.kth.se/energy-systems"><img src="https://avatars.githubusercontent.com/u/3727919?v=4?s=100" width="100px;" alt="Will Usher"/><br /><sub><b>Will Usher</b></sub></a><br /><a href="#code-willu47" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sgreenbury"><img src="https://avatars.githubusercontent.com/u/50113363?v=4?s=100" width="100px;" alt="Sam Greenbury"/><br /><sub><b>Sam Greenbury</b></sub></a><br /><a href="#code-sgreenbury" title="Code">💻</a> <a href="#ideas-sgreenbury" title="Ideas, Planning, & Feedback">🤔</a> <a href="#review-sgreenbury" title="Reviewed Pull Requests">👀</a> <a href="#projectManagement-sgreenbury" title="Project Management">📆</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://edchalstrey.com/"><img src="https://avatars.githubusercontent.com/u/5486164?v=4?s=100" width="100px;" alt="Ed Chalstrey"/><br /><sub><b>Ed Chalstrey</b></sub></a><br /><a href="#code-edwardchalstrey1" title="Code">💻</a> <a href="#design-edwardchalstrey1" title="Design">🎨</a> <a href="#review-edwardchalstrey1" title="Reviewed Pull Requests">👀</a> <a href="#doc-edwardchalstrey1" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/EdwinB12"><img src="https://avatars.githubusercontent.com/u/64434531?v=4?s=100" width="100px;" alt="Edwin "/><br /><sub><b>Edwin </b></sub></a><br /><a href="#code-EdwinB12" title="Code">💻</a> <a href="#ideas-EdwinB12" title="Ideas, Planning, & Feedback">🤔</a> <a href="#review-EdwinB12" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://paolo-conti.com/"><img src="https://avatars.githubusercontent.com/u/51111500?v=4?s=100" width="100px;" alt="Paolo Conti"/><br /><sub><b>Paolo Conti</b></sub></a><br /><a href="#code-ContiPaolo" title="Code">💻</a> <a href="#ideas-ContiPaolo" title="Ideas, Planning, & Feedback">🤔</a> <a href="#review-ContiPaolo" title="Reviewed Pull Requests">👀</a> <a href="#doc-ContiPaolo" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
