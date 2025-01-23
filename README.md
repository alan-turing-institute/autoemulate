# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                         |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| autoemulate/\_\_init\_\_.py                                                  |        0 |        0 |    100% |           |
| autoemulate/compare.py                                                       |      169 |       22 |     87% |240-244, 278, 315, 317, 320, 338, 342, 345, 359, 465, 521-532, 594-603, 625 |
| autoemulate/cross\_validate.py                                               |       44 |        7 |     84% |61-79, 150 |
| autoemulate/data\_splitting.py                                               |        6 |        0 |    100% |           |
| autoemulate/datasets.py                                                      |       12 |        0 |    100% |           |
| autoemulate/emulators/\_\_init\_\_.py                                        |       27 |        0 |    100% |           |
| autoemulate/emulators/conditional\_neural\_process.py                        |      102 |        3 |     97% |253-254, 295 |
| autoemulate/emulators/conditional\_neural\_process\_attn.py                  |        9 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process.py                                   |      104 |        5 |     95% |73, 266, 289, 321, 332 |
| autoemulate/emulators/gaussian\_process\_mogp.py                             |       33 |       18 |     45% |18, 35-40, 59-65, 71-75, 79, 82 |
| autoemulate/emulators/gaussian\_process\_mt.py                               |       97 |        7 |     93% |71, 240, 246, 265, 271, 286, 290 |
| autoemulate/emulators/gaussian\_process\_sklearn.py                          |       42 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process\_utils/\_\_init\_\_.py               |        3 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process\_utils/early\_stopping\_criterion.py |       11 |        2 |     82% |    58, 63 |
| autoemulate/emulators/gaussian\_process\_utils/poly\_mean.py                 |       23 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process\_utils/polynomial\_features.py       |       35 |        0 |    100% |           |
| autoemulate/emulators/gradient\_boosting.py                                  |       42 |        0 |    100% |           |
| autoemulate/emulators/light\_gbm.py                                          |       52 |        0 |    100% |           |
| autoemulate/emulators/neural\_net\_sk.py                                     |       42 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/\_\_init\_\_.py                       |        0 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/cnp\_module.py                        |       47 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/cnp\_module\_attn.py                  |       50 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/datasets.py                           |       49 |        1 |     98% |        11 |
| autoemulate/emulators/neural\_networks/gp\_module.py                         |       23 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/losses.py                             |       12 |        0 |    100% |           |
| autoemulate/emulators/polynomials.py                                         |       33 |        0 |    100% |           |
| autoemulate/emulators/radial\_basis\_functions.py                            |       34 |        0 |    100% |           |
| autoemulate/emulators/random\_forest.py                                      |       39 |        0 |    100% |           |
| autoemulate/emulators/support\_vector\_machines.py                           |       54 |        3 |     94% |     80-82 |
| autoemulate/experimental\_design.py                                          |       19 |        3 |     84% |24, 35, 46 |
| autoemulate/hyperparam\_searching.py                                         |       39 |        3 |     92% | 77, 82-83 |
| autoemulate/logging\_config.py                                               |       43 |        4 |     91% |28, 56, 63-64 |
| autoemulate/metrics.py                                                       |        8 |        0 |    100% |           |
| autoemulate/model\_processing.py                                             |       21 |        0 |    100% |           |
| autoemulate/model\_registry.py                                               |       31 |        1 |     97% |        46 |
| autoemulate/plotting.py                                                      |      163 |        6 |     96% |122, 132, 226, 403, 408, 418 |
| autoemulate/printing.py                                                      |       37 |       14 |     62% |7, 12, 17-26, 40, 97, 103-105 |
| autoemulate/save.py                                                          |       36 |        3 |     92% |     28-30 |
| autoemulate/sensitivity\_analysis.py                                         |      108 |       41 |     62% |44-49, 57, 60, 62, 67, 95, 127-130, 171, 227-241, 264-304 |
| autoemulate/simulations/\_\_init\_\_.py                                      |        0 |        0 |    100% |           |
| autoemulate/simulations/epidemic.py                                          |       26 |       26 |      0% |      1-55 |
| autoemulate/simulations/projectile.py                                        |       46 |        8 |     83% |177-182, 199-200, 221-223 |
| autoemulate/utils.py                                                         |      125 |        6 |     95% |57, 65, 180, 362-363, 379 |
| tests/\_\_init\_\_.py                                                        |        0 |        0 |    100% |           |
| tests/models/test\_attn\_cnp.py                                              |      147 |        0 |    100% |           |
| tests/models/test\_cnp.py                                                    |      108 |        0 |    100% |           |
| tests/models/test\_cnp\_dataset.py                                           |       74 |        0 |    100% |           |
| tests/models/test\_gptorch.py                                                |       66 |        4 |     94% |     62-65 |
| tests/test\_compare.py                                                       |      149 |        0 |    100% |           |
| tests/test\_cross\_validate.py                                               |       97 |        3 |     97% | 56-59, 64 |
| tests/test\_data\_splitting.py                                               |       11 |        0 |    100% |           |
| tests/test\_datasets.py                                                      |       13 |        0 |    100% |           |
| tests/test\_end\_to\_end.py                                                  |       39 |        0 |    100% |           |
| tests/test\_estimators.py                                                    |       25 |        0 |    100% |           |
| tests/test\_experimental\_design.py                                          |       22 |        0 |    100% |           |
| tests/test\_gaussian\_process\_utils.py                                      |       76 |        0 |    100% |           |
| tests/test\_hyperparam\_searching.py                                         |       48 |        0 |    100% |           |
| tests/test\_logging\_config.py                                               |       51 |        0 |    100% |           |
| tests/test\_model\_processing.py                                             |       54 |        0 |    100% |           |
| tests/test\_model\_registry.py                                               |       86 |        0 |    100% |           |
| tests/test\_plotting.py                                                      |      194 |        7 |     96% |44, 54, 76-77, 85-86, 94 |
| tests/test\_printing.py                                                      |       19 |        0 |    100% |           |
| tests/test\_save.py                                                          |       62 |        2 |     97% |    30, 35 |
| tests/test\_sensitivity\_analysis.py                                         |      111 |        0 |    100% |           |
| tests/test\_ui.py                                                            |       57 |        0 |    100% |           |
| tests/test\_utils.py                                                         |      182 |        6 |     97% |51, 57, 62, 67, 72, 77 |
|                                                                    **TOTAL** | **3587** |  **205** | **94%** |           |


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