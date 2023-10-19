# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                        |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------------- | -------: | -------: | ------: | --------: |
| autoemulate/\_\_init\_\_.py                 |        0 |        0 |    100% |           |
| autoemulate/compare.py                      |      112 |       48 |     57% |53, 72-96, 219-273 |
| autoemulate/cv.py                           |        6 |        1 |     83% |        37 |
| autoemulate/emulators/\_\_init\_\_.py       |        6 |        0 |    100% |           |
| autoemulate/emulators/base.py               |       14 |        4 |     71% |10, 24, 38, 53 |
| autoemulate/emulators/gaussian\_process2.py |       15 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process.py  |       19 |        0 |    100% |           |
| autoemulate/emulators/neural\_network.py    |       19 |        1 |     95% |        58 |
| autoemulate/emulators/random\_forest.py     |       15 |        0 |    100% |           |
| autoemulate/experimental\_design.py         |       18 |        3 |     83% |16, 27, 38 |
| autoemulate/metrics.py                      |        7 |        0 |    100% |           |
| tests/\_\_init\_\_.py                       |        0 |        0 |    100% |           |
| tests/test\_compare.py                      |       65 |        0 |    100% |           |
| tests/test\_emulators.py                    |       63 |        0 |    100% |           |
| tests/test\_experimental\_design.py         |       21 |        0 |    100% |           |
|                                   **TOTAL** |  **380** |   **57** | **85%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/alan-turing-institute/autoemulate/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/alan-turing-institute/autoemulate/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Falan-turing-institute%2Fautoemulate%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.