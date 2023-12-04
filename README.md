# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------- | -------: | -------: | ------: | --------: |
| autoemulate/\_\_init\_\_.py                         |        0 |        0 |    100% |           |
| autoemulate/compare.py                              |      121 |       16 |     87% |247, 308-310, 351-378, 452 |
| autoemulate/cv.py                                   |        4 |        0 |    100% |           |
| autoemulate/datasets.py                             |       20 |        6 |     70% |     89-96 |
| autoemulate/emulators/\_\_init\_\_.py               |        9 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process.py          |       26 |        3 |     88% | 67-68, 71 |
| autoemulate/emulators/gaussian\_process\_sk.py      |       31 |        2 |     94% |     87-99 |
| autoemulate/emulators/gradient\_boosting.py         |       32 |        0 |    100% |           |
| autoemulate/emulators/neural\_net\_sk.py            |       35 |        2 |     94% |    93-109 |
| autoemulate/emulators/neural\_net\_torch.py         |       35 |       19 |     46% |17-18, 24-29, 32-35, 38-42, 65, 84, 97 |
| autoemulate/emulators/radial\_basis.py              |       29 |        2 |     93% |     75-80 |
| autoemulate/emulators/random\_forest.py             |       32 |        0 |    100% |           |
| autoemulate/emulators/second\_order\_polynomials.py |       24 |        1 |     96% |        68 |
| autoemulate/experimental\_design.py                 |       18 |        3 |     83% |16, 27, 38 |
| autoemulate/hyperparam\_search.py                   |       48 |       10 |     79% |72, 79, 93-97, 120-125 |
| autoemulate/logging\_config.py                      |       21 |        4 |     81% |     24-27 |
| autoemulate/metrics.py                              |        7 |        0 |    100% |           |
| autoemulate/plotting.py                             |       38 |       32 |     16% |22-38, 58-59, 85-92, 116-129 |
| autoemulate/utils.py                                |       56 |        5 |     91% |27, 89, 94-99 |
| tests/\_\_init\_\_.py                               |        0 |        0 |    100% |           |
| tests/test\_compare.py                              |      110 |        0 |    100% |           |
| tests/test\_datasets.py                             |       17 |        0 |    100% |           |
| tests/test\_emulators.py                            |       63 |        0 |    100% |           |
| tests/test\_estimators.py                           |       12 |        0 |    100% |           |
| tests/test\_experimental\_design.py                 |       21 |        0 |    100% |           |
| tests/test\_hyperparam\_search.py                   |       36 |        0 |    100% |           |
| tests/test\_utils.py                                |       91 |        6 |     93% |50, 56, 61, 66, 71, 76 |
|                                           **TOTAL** |  **936** |  **111** | **88%** |           |


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