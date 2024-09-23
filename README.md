# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                   |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------- | -------: | -------: | ------: | --------: |
| autoemulate/\_\_init\_\_.py                            |        0 |        0 |    100% |           |
| autoemulate/compare.py                                 |      162 |       23 |     86% |243-247, 287, 326, 340, 356-364, 375-378, 382-386, 390, 522-533 |
| autoemulate/cross\_validate.py                         |       29 |        6 |     79% |     58-76 |
| autoemulate/data\_splitting.py                         |        6 |        0 |    100% |           |
| autoemulate/datasets.py                                |       12 |        0 |    100% |           |
| autoemulate/emulators/\_\_init\_\_.py                  |       23 |        0 |    100% |           |
| autoemulate/emulators/conditional\_neural\_process.py  |      102 |        8 |     92% |141-142, 254-255, 261-289, 294, 308 |
| autoemulate/emulators/gaussian\_process\_mogp.py       |       39 |       22 |     44% |20, 37-42, 61-67, 71-83, 87, 90 |
| autoemulate/emulators/gaussian\_process\_sklearn.py    |       49 |        7 |     86% |    99-122 |
| autoemulate/emulators/gaussian\_process\_torch.py      |       72 |        3 |     96% |72, 194-209 |
| autoemulate/emulators/gradient\_boosting.py            |       49 |        0 |    100% |           |
| autoemulate/emulators/light\_gbm.py                    |       59 |        7 |     88% |   110-137 |
| autoemulate/emulators/neural\_net\_sk.py               |       48 |        7 |     85% |   101-135 |
| autoemulate/emulators/neural\_networks/\_\_init\_\_.py |        0 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/cnp\_module.py  |       49 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/datasets.py     |       49 |        1 |     98% |        11 |
| autoemulate/emulators/neural\_networks/gp\_module.py   |       14 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/losses.py       |       12 |        0 |    100% |           |
| autoemulate/emulators/polynomials.py                   |       37 |        5 |     86% |     84-89 |
| autoemulate/emulators/radial\_basis\_functions.py      |       41 |        7 |     83% |    94-145 |
| autoemulate/emulators/random\_forest.py                |       46 |        2 |     96% |   123-124 |
| autoemulate/emulators/support\_vector\_machines.py     |       61 |       10 |     84% |83-85, 128-161 |
| autoemulate/experimental\_design.py                    |       19 |        3 |     84% |24, 35, 46 |
| autoemulate/hyperparam\_searching.py                   |       45 |        7 |     84% |81-96, 101-102 |
| autoemulate/logging\_config.py                         |       44 |        4 |     91% |29, 57, 64-65 |
| autoemulate/metrics.py                                 |        8 |        0 |    100% |           |
| autoemulate/model\_processing.py                       |       23 |        0 |    100% |           |
| autoemulate/model\_registry.py                         |       31 |        1 |     97% |        44 |
| autoemulate/plotting.py                                |      169 |        7 |     96% |123, 138-139, 230, 407, 412, 422 |
| autoemulate/printing.py                                |       54 |       10 |     81% |10, 15, 23-26, 73, 129-131, 154 |
| autoemulate/save.py                                    |       48 |        4 |     92% |63-65, 90, 104 |
| autoemulate/utils.py                                   |      142 |       10 |     93% |37, 56, 64, 148, 216, 221-226, 472-473 |
| tests/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| tests/models/test\_cnp.py                              |      108 |        0 |    100% |           |
| tests/models/test\_cnp\_dataset.py                     |       74 |        0 |    100% |           |
| tests/models/test\_gptorch.py                          |       33 |        0 |    100% |           |
| tests/test\_compare.py                                 |      126 |        0 |    100% |           |
| tests/test\_cross\_validate.py                         |       49 |        0 |    100% |           |
| tests/test\_data\_splitting.py                         |       11 |        0 |    100% |           |
| tests/test\_datasets.py                                |       13 |        0 |    100% |           |
| tests/test\_estimators.py                              |       23 |        0 |    100% |           |
| tests/test\_experimental\_design.py                    |       22 |        0 |    100% |           |
| tests/test\_hyperparam\_searching.py                   |       48 |        0 |    100% |           |
| tests/test\_logging\_config.py                         |       51 |        0 |    100% |           |
| tests/test\_model\_processing.py                       |       54 |        0 |    100% |           |
| tests/test\_model\_registry.py                         |       81 |        0 |    100% |           |
| tests/test\_plotting.py                                |      194 |        7 |     96% |44, 54, 76-77, 85-86, 94 |
| tests/test\_printing.py                                |       46 |        0 |    100% |           |
| tests/test\_save.py                                    |       85 |        1 |     99% |        32 |
| tests/test\_ui.py                                      |       39 |        0 |    100% |           |
| tests/test\_utils.py                                   |      204 |        6 |     97% |50, 56, 61, 66, 71, 76 |
|                                              **TOTAL** | **2803** |  **168** | **94%** |           |


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