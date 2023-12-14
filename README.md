# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                               |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------- | -------: | -------: | ------: | --------: |
| autoemulate/\_\_init\_\_.py                        |        0 |        0 |    100% |           |
| autoemulate/compare.py                             |      131 |       21 |     84% |248, 310-312, 358-383, 428-431, 435-436, 471 |
| autoemulate/cv.py                                  |        4 |        0 |    100% |           |
| autoemulate/datasets.py                            |       20 |        6 |     70% |     89-96 |
| autoemulate/emulators/\_\_init\_\_.py              |       11 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process.py         |       32 |        8 |     75% | 67-79, 82 |
| autoemulate/emulators/gaussian\_process\_sk.py     |       38 |        7 |     82% |    91-114 |
| autoemulate/emulators/gradient\_boosting.py        |       39 |        0 |    100% |           |
| autoemulate/emulators/neural\_net\_sk.py           |       41 |        7 |     83% |    97-131 |
| autoemulate/emulators/neural\_net\_torch.py        |       42 |       25 |     40% |18-19, 25-30, 33-36, 39-43, 66, 85-107, 110 |
| autoemulate/emulators/polynomials.py               |       30 |        5 |     83% |     82-87 |
| autoemulate/emulators/random\_forest.py            |       38 |        2 |     95% |   120-121 |
| autoemulate/emulators/rbf.py                       |       32 |        7 |     78% |    90-141 |
| autoemulate/emulators/support\_vector\_machines.py |       51 |       10 |     80% |78-80, 123-156 |
| autoemulate/emulators/xgboost.py                   |       49 |        7 |     86% |   130-167 |
| autoemulate/experimental\_design.py                |       18 |        3 |     83% |16, 27, 38 |
| autoemulate/hyperparam\_search.py                  |       42 |        4 |     90% |74, 82, 97-98 |
| autoemulate/logging\_config.py                     |       21 |        4 |     81% |     24-27 |
| autoemulate/metrics.py                             |        7 |        0 |    100% |           |
| autoemulate/plotting.py                            |       38 |       32 |     16% |22-38, 58-59, 85-92, 116-129 |
| autoemulate/save.py                                |       29 |        1 |     97% |        44 |
| autoemulate/utils.py                               |       77 |        5 |     94% |29, 91, 96-101 |
| tests/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| tests/test\_compare.py                             |      109 |        0 |    100% |           |
| tests/test\_datasets.py                            |       17 |        0 |    100% |           |
| tests/test\_emulators.py                           |       63 |        0 |    100% |           |
| tests/test\_estimators.py                          |        6 |        0 |    100% |           |
| tests/test\_experimental\_design.py                |       21 |        0 |    100% |           |
| tests/test\_hyperparam\_search.py                  |       43 |        0 |    100% |           |
| tests/test\_save.py                                |       42 |        0 |    100% |           |
| tests/test\_utils.py                               |      141 |        6 |     96% |58, 64, 69, 74, 79, 84 |
|                                          **TOTAL** | **1232** |  **160** | **87%** |           |


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