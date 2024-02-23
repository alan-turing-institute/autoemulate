# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                   |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------- | -------: | -------: | ------: | --------: |
| autoemulate/\_\_init\_\_.py                            |        0 |        0 |    100% |           |
| autoemulate/compare.py                                 |      133 |       24 |     82% |202, 224-227, 303, 317, 333-341, 352-355, 359-363, 376, 409, 466 |
| autoemulate/cross\_validate.py                         |       25 |        3 |     88% |     32-34 |
| autoemulate/cv.py                                      |        5 |        0 |    100% |           |
| autoemulate/data\_splitting.py                         |        6 |        0 |    100% |           |
| autoemulate/datasets.py                                |       12 |        0 |    100% |           |
| autoemulate/emulators/\_\_init\_\_.py                  |       11 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process.py             |       36 |        8 |     78% | 71-83, 86 |
| autoemulate/emulators/gaussian\_process\_sk.py         |       46 |        7 |     85% |    98-121 |
| autoemulate/emulators/gradient\_boosting.py            |       46 |        0 |    100% |           |
| autoemulate/emulators/neural\_net\_sk.py               |       45 |        7 |     84% |   101-135 |
| autoemulate/emulators/neural\_net\_torch.py            |       99 |        3 |     97% |93, 107, 146 |
| autoemulate/emulators/neural\_networks/\_\_init\_\_.py |        2 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/base.py         |       15 |        2 |     87% |    28, 31 |
| autoemulate/emulators/neural\_networks/get\_module.py  |       11 |        2 |     82% |     15-16 |
| autoemulate/emulators/neural\_networks/mlp.py          |       31 |        9 |     71% |     41-63 |
| autoemulate/emulators/polynomials.py                   |       34 |        5 |     85% |     84-89 |
| autoemulate/emulators/random\_forest.py                |       43 |        2 |     95% |   123-124 |
| autoemulate/emulators/rbf.py                           |       38 |        7 |     82% |    94-145 |
| autoemulate/emulators/support\_vector\_machines.py     |       58 |       10 |     83% |83-85, 128-161 |
| autoemulate/emulators/xgboost.py                       |       56 |        7 |     88% |   135-172 |
| autoemulate/experimental\_design.py                    |       19 |        3 |     84% |18, 29, 40 |
| autoemulate/hyperparam\_searching.py                   |       45 |        8 |     82% |70-82, 87-91 |
| autoemulate/logging\_config.py                         |       22 |        4 |     82% |     25-28 |
| autoemulate/metrics.py                                 |        8 |        0 |    100% |           |
| autoemulate/model\_processing.py                       |       31 |        0 |    100% |           |
| autoemulate/plotting.py                                |       84 |       55 |     35% |128-151, 188-210, 247-262, 288-334 |
| autoemulate/printing.py                                |       15 |        0 |    100% |           |
| autoemulate/save.py                                    |       39 |        2 |     95% |    38, 61 |
| autoemulate/utils.py                                   |      100 |        7 |     93% |32, 94, 99-104, 330-331 |
| tests/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| tests/test\_compare.py                                 |      104 |        0 |    100% |           |
| tests/test\_cross\_validate.py                         |       48 |        0 |    100% |           |
| tests/test\_data\_splitting.py                         |       11 |        0 |    100% |           |
| tests/test\_datasets.py                                |       13 |        0 |    100% |           |
| tests/test\_emulators.py                               |      135 |        0 |    100% |           |
| tests/test\_estimators.py                              |       16 |        0 |    100% |           |
| tests/test\_experimental\_design.py                    |       22 |        0 |    100% |           |
| tests/test\_hyperparam\_searching.py                   |       48 |        0 |    100% |           |
| tests/test\_model\_processing.py                       |       58 |        0 |    100% |           |
| tests/test\_plotting.py                                |       68 |        7 |     90% |18, 28, 50-51, 59-60, 68 |
| tests/test\_printing.py                                |       32 |        0 |    100% |           |
| tests/test\_save.py                                    |       88 |        0 |    100% |           |
| tests/test\_utils.py                                   |      167 |        6 |     96% |58, 64, 69, 74, 79, 84 |
|                                              **TOTAL** | **1925** |  **188** | **90%** |           |


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