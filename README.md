# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                   |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------- | -------: | -------: | ------: | --------: |
| autoemulate/\_\_init\_\_.py                            |        0 |        0 |    100% |           |
| autoemulate/compare.py                                 |      143 |       21 |     85% |229-231, 307, 321, 337-345, 356-359, 363-367, 424, 481 |
| autoemulate/cross\_validate.py                         |       24 |        2 |     92% |     56-57 |
| autoemulate/data\_splitting.py                         |        6 |        0 |    100% |           |
| autoemulate/datasets.py                                |       12 |        0 |    100% |           |
| autoemulate/emulators/\_\_init\_\_.py                  |       11 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process.py             |       49 |        7 |     86% |    99-122 |
| autoemulate/emulators/gaussian\_process\_mogp.py       |       39 |        9 |     77% |71-83, 87, 90 |
| autoemulate/emulators/gradient\_boosting.py            |       49 |        0 |    100% |           |
| autoemulate/emulators/light\_gbm.py                    |       59 |        7 |     88% |   110-137 |
| autoemulate/emulators/neural\_net\_sk.py               |       48 |        7 |     85% |   101-135 |
| autoemulate/emulators/neural\_net\_torch.py            |       99 |        1 |     99% |       126 |
| autoemulate/emulators/neural\_networks/\_\_init\_\_.py |        2 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/base.py         |       19 |        3 |     84% |28, 33, 36 |
| autoemulate/emulators/neural\_networks/get\_module.py  |       14 |        2 |     86% |     18-19 |
| autoemulate/emulators/neural\_networks/mlp.py          |       32 |        4 |     88% |     59-62 |
| autoemulate/emulators/neural\_networks/rbf.py          |       97 |       19 |     80% |24, 28, 32, 36, 138, 150, 160, 257, 262, 267, 318-340 |
| autoemulate/emulators/polynomials.py                   |       37 |        5 |     86% |     84-89 |
| autoemulate/emulators/random\_forest.py                |       46 |        2 |     96% |   123-124 |
| autoemulate/emulators/rbf.py                           |       41 |        7 |     83% |    94-145 |
| autoemulate/emulators/support\_vector\_machines.py     |       61 |       10 |     84% |83-85, 128-161 |
| autoemulate/experimental\_design.py                    |       19 |        3 |     84% |24, 35, 46 |
| autoemulate/hyperparam\_searching.py                   |       43 |        7 |     84% |70-83, 88-89 |
| autoemulate/logging\_config.py                         |       44 |        4 |     91% |29, 57, 64-65 |
| autoemulate/metrics.py                                 |        8 |        0 |    100% |           |
| autoemulate/model\_processing.py                       |       33 |        0 |    100% |           |
| autoemulate/plotting.py                                |       91 |       57 |     37% |96-97, 137-160, 196-218, 255-270, 294-340 |
| autoemulate/printing.py                                |       48 |        9 |     81% |9, 14, 22-25, 72, 128-130 |
| autoemulate/save.py                                    |       40 |        3 |     92% | 56-57, 80 |
| autoemulate/utils.py                                   |      100 |        9 |     91% |32, 91, 97, 125, 130-135, 381-382 |
| tests/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| tests/test\_compare.py                                 |      103 |        0 |    100% |           |
| tests/test\_cross\_validate.py                         |       49 |        0 |    100% |           |
| tests/test\_data\_splitting.py                         |       11 |        0 |    100% |           |
| tests/test\_datasets.py                                |       13 |        0 |    100% |           |
| tests/test\_emulators.py                               |       86 |        0 |    100% |           |
| tests/test\_estimators.py                              |       16 |        0 |    100% |           |
| tests/test\_experimental\_design.py                    |       22 |        0 |    100% |           |
| tests/test\_hyperparam\_searching.py                   |       48 |        0 |    100% |           |
| tests/test\_logging\_config.py                         |       51 |        0 |    100% |           |
| tests/test\_model\_processing.py                       |       67 |        0 |    100% |           |
| tests/test\_plotting.py                                |       68 |        7 |     90% |18, 28, 50-51, 59-60, 68 |
| tests/test\_printing.py                                |       30 |        0 |    100% |           |
| tests/test\_save.py                                    |       88 |        0 |    100% |           |
| tests/test\_torch.py                                   |       95 |        0 |    100% |           |
| tests/test\_ui.py                                      |       39 |        0 |    100% |           |
| tests/test\_utils.py                                   |      164 |        6 |     96% |53, 59, 64, 69, 74, 79 |
|                                              **TOTAL** | **2264** |  **211** | **91%** |           |


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