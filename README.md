# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                           |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------- | -------: | -------: | ------: | --------: |
| autoemulate/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| autoemulate/compare.py                         |       79 |       22 |     72% |88, 100, 196-204, 208-228, 239 |
| autoemulate/cv.py                              |        6 |        1 |     83% |        37 |
| autoemulate/datasets.py                        |       13 |        0 |    100% |           |
| autoemulate/emulators/\_\_init\_\_.py          |        8 |        0 |    100% |           |
| autoemulate/emulators/base.py                  |       14 |        4 |     71% |10, 24, 38, 53 |
| autoemulate/emulators/gaussian\_process.py     |       27 |        3 |     89% | 67-68, 71 |
| autoemulate/emulators/gaussian\_process\_sk.py |       31 |        2 |     94% |    89-100 |
| autoemulate/emulators/neural\_net\_sk.py       |       33 |        2 |     94% |    92-101 |
| autoemulate/emulators/neural\_net\_torch.py    |       34 |       19 |     44% |16-17, 23-28, 31-34, 37-41, 64, 83, 90 |
| autoemulate/emulators/radial\_basis.py         |       29 |        2 |     93% |     75-80 |
| autoemulate/emulators/random\_forest.py        |       33 |        2 |     94% |    95-104 |
| autoemulate/experimental\_design.py            |       18 |        3 |     83% |16, 27, 38 |
| autoemulate/logging\_config.py                 |       21 |        4 |     81% |     24-27 |
| autoemulate/metrics.py                         |        7 |        0 |    100% |           |
| autoemulate/plotting.py                        |       38 |       32 |     16% |22-38, 58-59, 85-92, 116-129 |
| tests/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| tests/test\_compare.py                         |       39 |        0 |    100% |           |
| tests/test\_datasets.py                        |       17 |        0 |    100% |           |
| tests/test\_emulators.py                       |       63 |        0 |    100% |           |
| tests/test\_estimators.py                      |        6 |        0 |    100% |           |
| tests/test\_experimental\_design.py            |       21 |        0 |    100% |           |
|                                      **TOTAL** |  **537** |   **96** | **82%** |           |


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