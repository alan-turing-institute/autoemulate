# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                         |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| autoemulate/\_\_init\_\_.py                                                  |        0 |        0 |    100% |           |
| autoemulate/compare.py                                                       |      169 |       22 |     87% |240-244, 278, 315, 317, 320, 338, 342, 345, 359, 465, 521-532, 594-602, 624 |
| autoemulate/cross\_validate.py                                               |       44 |        7 |     84% |61-79, 150 |
| autoemulate/data\_splitting.py                                               |        6 |        0 |    100% |           |
| autoemulate/datasets.py                                                      |       12 |        0 |    100% |           |
| autoemulate/emulators/\_\_init\_\_.py                                        |       27 |        0 |    100% |           |
| autoemulate/emulators/conditional\_neural\_process.py                        |      103 |        3 |     97% |254-255, 296 |
| autoemulate/emulators/conditional\_neural\_process\_attn.py                  |        9 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process.py                                   |      104 |        9 |     91% |73, 259, 266, 289, 300, 310, 321, 340, 344 |
| autoemulate/emulators/gaussian\_process\_mogp.py                             |       33 |       18 |     45% |18, 35-40, 59-65, 71-75, 79, 82 |
| autoemulate/emulators/gaussian\_process\_mt.py                               |       97 |        9 |     91% |71, 240, 246, 250, 254, 265, 271, 282, 286 |
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
| autoemulate/experimental/data/preprocessors.py                               |       23 |        3 |     87% |18, 20, 38 |
| autoemulate/experimental/data/utils.py                                       |       35 |        3 |     91% |53, 80, 84 |
| autoemulate/experimental/emulators/\_\_init\_\_.py                           |        0 |        0 |    100% |           |
| autoemulate/experimental/emulators/base.py                                   |       59 |        6 |     90% |20, 27, 54, 121, 129, 133 |
| autoemulate/experimental/tuner.py                                            |       38 |        8 |     79% |61-63, 75-84 |
| autoemulate/experimental/types.py                                            |       13 |        0 |    100% |           |
| autoemulate/experimental\_design.py                                          |       18 |        3 |     83% |23, 34, 45 |
| autoemulate/history\_matching.py                                             |       18 |        2 |     89% |    30, 34 |
| autoemulate/hyperparam\_searching.py                                         |       39 |        3 |     92% | 77, 82-83 |
| autoemulate/logging\_config.py                                               |       43 |        4 |     91% |28, 56, 63-64 |
| autoemulate/metrics.py                                                       |        7 |        0 |    100% |           |
| autoemulate/model\_processing.py                                             |       21 |        0 |    100% |           |
| autoemulate/model\_registry.py                                               |       31 |        1 |     97% |        46 |
| autoemulate/plotting.py                                                      |      177 |        9 |     95% |146, 156, 250, 423, 428, 438, 498, 618-619 |
| autoemulate/printing.py                                                      |       37 |       14 |     62% |7, 12, 17-26, 40, 97, 103-105 |
| autoemulate/save.py                                                          |       36 |        3 |     92% |     28-30 |
| autoemulate/sensitivity\_analysis.py                                         |      111 |       39 |     65% |48-53, 61, 64, 66, 71, 99, 133-136, 235-249, 272-310 |
| autoemulate/simulations/\_\_init\_\_.py                                      |        0 |        0 |    100% |           |
| autoemulate/simulations/epidemic.py                                          |       26 |       26 |      0% |      1-55 |
| autoemulate/simulations/flow\_functions.py                                   |       85 |       85 |      0% |     1-162 |
| autoemulate/simulations/projectile.py                                        |       46 |        8 |     83% |177-182, 199-200, 221-223 |
| autoemulate/utils.py                                                         |      146 |        9 |     94% |57, 65, 180, 362-363, 379, 426, 435, 449 |
| tests/\_\_init\_\_.py                                                        |        0 |        0 |    100% |           |
| tests/experimental/test\_experimental\_base.py                               |       95 |        0 |    100% |           |
| tests/models/test\_attn\_cnp.py                                              |      147 |        0 |    100% |           |
| tests/models/test\_cnp.py                                                    |      108 |        0 |    100% |           |
| tests/models/test\_cnp\_dataset.py                                           |       74 |        0 |    100% |           |
| tests/models/test\_gptorch.py                                                |       81 |        5 |     94% |     77-81 |
| tests/test\_compare.py                                                       |      149 |        0 |    100% |           |
| tests/test\_cross\_validate.py                                               |       97 |        3 |     97% | 56-59, 64 |
| tests/test\_data\_splitting.py                                               |       11 |        0 |    100% |           |
| tests/test\_datasets.py                                                      |       13 |        0 |    100% |           |
| tests/test\_end\_to\_end.py                                                  |       39 |        0 |    100% |           |
| tests/test\_estimators.py                                                    |       26 |        0 |    100% |           |
| tests/test\_experimental\_design.py                                          |       22 |        0 |    100% |           |
| tests/test\_gaussian\_process\_utils.py                                      |       76 |        0 |    100% |           |
| tests/test\_history\_matching.py                                             |       43 |        0 |    100% |           |
| tests/test\_hyperparam\_searching.py                                         |       48 |        0 |    100% |           |
| tests/test\_logging\_config.py                                               |       51 |        0 |    100% |           |
| tests/test\_model\_processing.py                                             |       54 |        0 |    100% |           |
| tests/test\_model\_registry.py                                               |       86 |        0 |    100% |           |
| tests/test\_plotting.py                                                      |      205 |        7 |     97% |44, 54, 76-77, 85-86, 94 |
| tests/test\_printing.py                                                      |       19 |        0 |    100% |           |
| tests/test\_pytorch\_utils.py                                                |       63 |        0 |    100% |           |
| tests/test\_save.py                                                          |       62 |        2 |     97% |    30, 35 |
| tests/test\_sensitivity\_analysis.py                                         |      116 |        0 |    100% |           |
| tests/test\_ui.py                                                            |       55 |        0 |    100% |           |
| tests/test\_utils.py                                                         |      182 |        6 |     97% |51, 57, 62, 67, 72, 77 |
|                                                                    **TOTAL** | **4126** |  **323** | **92%** |           |


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