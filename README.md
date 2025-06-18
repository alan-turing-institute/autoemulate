# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                                 |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| autoemulate/\_\_init\_\_.py                                                          |        0 |        0 |    100% |           |
| autoemulate/compare.py                                                               |      340 |       80 |     76% |267-277, 333-341, 430-440, 447, 451-459, 471, 488, 495-500, 506, 566, 582, 584, 587, 605, 609, 612, 626, 653, 658-660, 667, 672, 680, 690, 749, 755-756, 765-772, 785, 798-799, 823, 858, 864, 950-980, 1049-1071, 1095-1098 |
| autoemulate/cross\_validate.py                                                       |       46 |        3 |     93% |81-82, 153 |
| autoemulate/data\_splitting.py                                                       |        6 |        0 |    100% |           |
| autoemulate/datasets.py                                                              |       12 |        0 |    100% |           |
| autoemulate/emulators/\_\_init\_\_.py                                                |       27 |        0 |    100% |           |
| autoemulate/emulators/conditional\_neural\_process.py                                |      103 |        3 |     97% |254-255, 296 |
| autoemulate/emulators/conditional\_neural\_process\_attn.py                          |        9 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process.py                                           |      104 |        2 |     98% |   73, 282 |
| autoemulate/emulators/gaussian\_process\_mogp.py                                     |       33 |       18 |     45% |18, 35-40, 59-65, 71-75, 79, 82 |
| autoemulate/emulators/gaussian\_process\_mt.py                                       |       97 |       10 |     90% |71, 246, 250, 254, 259, 265, 271, 282, 286, 290 |
| autoemulate/emulators/gaussian\_process\_sklearn.py                                  |       42 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process\_utils/\_\_init\_\_.py                       |        3 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process\_utils/early\_stopping\_criterion.py         |       11 |        2 |     82% |    58, 63 |
| autoemulate/emulators/gaussian\_process\_utils/poly\_mean.py                         |       23 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process\_utils/polynomial\_features.py               |       35 |        0 |    100% |           |
| autoemulate/emulators/gradient\_boosting.py                                          |       42 |        0 |    100% |           |
| autoemulate/emulators/light\_gbm.py                                                  |       52 |        0 |    100% |           |
| autoemulate/emulators/neural\_net\_sk.py                                             |       42 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/\_\_init\_\_.py                               |        0 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/cnp\_module.py                                |       47 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/cnp\_module\_attn.py                          |       50 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/datasets.py                                   |       49 |        1 |     98% |        11 |
| autoemulate/emulators/neural\_networks/gp\_module.py                                 |       23 |        0 |    100% |           |
| autoemulate/emulators/neural\_networks/losses.py                                     |       12 |        0 |    100% |           |
| autoemulate/emulators/polynomials.py                                                 |       33 |        0 |    100% |           |
| autoemulate/emulators/radial\_basis\_functions.py                                    |       34 |        0 |    100% |           |
| autoemulate/emulators/random\_forest.py                                              |       39 |        0 |    100% |           |
| autoemulate/emulators/support\_vector\_machines.py                                   |       54 |        3 |     94% |     80-82 |
| autoemulate/experimental/\_\_init\_\_.py                                             |        0 |        0 |    100% |           |
| autoemulate/experimental/compare.py                                                  |       54 |        0 |    100% |           |
| autoemulate/experimental/data/preprocessors.py                                       |       25 |        0 |    100% |           |
| autoemulate/experimental/data/utils.py                                               |      168 |       18 |     89% |63, 92, 98, 114-118, 132, 165, 170, 383, 386-388, 414, 417-419, 449 |
| autoemulate/experimental/device.py                                                   |       45 |        8 |     82% |15-16, 45, 99, 105, 154-159, 164 |
| autoemulate/experimental/emulators/\_\_init\_\_.py                                   |        6 |        0 |    100% |           |
| autoemulate/experimental/emulators/base.py                                           |      114 |        8 |     93% |47, 87-91, 193, 220-222, 235 |
| autoemulate/experimental/emulators/gaussian\_process/\_\_init\_\_.py                 |        8 |        0 |    100% |           |
| autoemulate/experimental/emulators/gaussian\_process/exact.py                        |       87 |        1 |     99% |        96 |
| autoemulate/experimental/emulators/lightgbm.py                                       |       47 |        0 |    100% |           |
| autoemulate/experimental/emulators/neural\_processes/conditional\_neural\_process.py |      170 |        7 |     96% |47-48, 193-194, 435, 464-465 |
| autoemulate/experimental/emulators/random\_forest.py                                 |       26 |        0 |    100% |           |
| autoemulate/experimental/emulators/svm.py                                            |       33 |        0 |    100% |           |
| autoemulate/experimental/learners/\_\_init\_\_.py                                    |        4 |        0 |    100% |           |
| autoemulate/experimental/learners/base.py                                            |      107 |       22 |     79% |61-64, 75-79, 92-101, 120, 125-129 |
| autoemulate/experimental/learners/membership.py                                      |        8 |        8 |      0% |      1-19 |
| autoemulate/experimental/learners/pool.py                                            |        8 |        8 |      0% |      1-19 |
| autoemulate/experimental/learners/stream.py                                          |      128 |       12 |     91% |83-93, 480, 531-535, 538, 566-571, 574 |
| autoemulate/experimental/model\_selection.py                                         |       45 |        3 |     93% |     30-37 |
| autoemulate/experimental/simulations/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| autoemulate/experimental/simulations/base.py                                         |       57 |        1 |     98% |        40 |
| autoemulate/experimental/simulations/epidemic.py                                     |       14 |       14 |      0% |      1-39 |
| autoemulate/experimental/simulations/projectile.py                                   |       24 |       24 |      0% |      1-76 |
| autoemulate/experimental/tuner.py                                                    |       32 |        0 |    100% |           |
| autoemulate/experimental/types.py                                                    |       18 |        0 |    100% |           |
| autoemulate/experimental\_design.py                                                  |       19 |        3 |     84% |24, 35, 46 |
| autoemulate/history\_matching.py                                                     |      115 |       41 |     64% |67, 71, 132-134, 206, 210-217, 230, 280, 293-295, 304-306, 326, 334-357, 404-432 |
| autoemulate/history\_matching\_dashboard.py                                          |      509 |      509 |      0% |    1-1216 |
| autoemulate/hyperparam\_searching.py                                                 |       46 |        3 |     93% |93, 99-100 |
| autoemulate/logging\_config.py                                                       |       43 |        4 |     91% |28, 56, 63-64 |
| autoemulate/mcmc.py                                                                  |      160 |      160 |      0% |     1-348 |
| autoemulate/mcmc\_dashboard.py                                                       |      369 |      369 |      0% |     1-816 |
| autoemulate/metrics.py                                                               |        7 |        0 |    100% |           |
| autoemulate/model\_processing.py                                                     |       48 |        1 |     98% |        67 |
| autoemulate/model\_registry.py                                                       |       31 |        1 |     97% |        46 |
| autoemulate/plotting.py                                                              |      179 |       10 |     94% |52, 145, 155, 249, 423, 428, 438, 498, 618-619 |
| autoemulate/preprocess\_target.py                                                    |      260 |       28 |     89% |84, 96, 116, 176, 208-209, 216, 256, 281, 319, 449-470, 485, 490, 520, 523, 526, 566, 607 |
| autoemulate/printing.py                                                              |       41 |       14 |     66% |7, 12, 17-26, 40, 128, 137-139 |
| autoemulate/save.py                                                                  |       36 |        3 |     92% |     28-30 |
| autoemulate/sensitivity\_analysis.py                                                 |      214 |      135 |     37% |54-76, 84, 87, 89, 94, 122, 156-159, 258-272, 295-333, 363-386, 407-423, 442-542, 551-623 |
| autoemulate/simulations/\_\_init\_\_.py                                              |        0 |        0 |    100% |           |
| autoemulate/simulations/base.py                                                      |       81 |       16 |     80% |60, 78, 103, 114, 136-139, 157, 172-188, 211-217 |
| autoemulate/simulations/circ\_utils.py                                               |       94 |       94 |      0% |     4-233 |
| autoemulate/simulations/epidemic.py                                                  |       35 |        0 |    100% |           |
| autoemulate/simulations/flow\_functions.py                                           |       85 |       85 |      0% |     1-162 |
| autoemulate/simulations/naghavi\_cardiac\_ModularCirc.py                             |       70 |       70 |      0% |     1-148 |
| autoemulate/simulations/projectile.py                                                |       55 |        8 |     85% |207-212, 229-230, 251-253 |
| autoemulate/simulations/reaction\_diffusion.py                                       |       57 |       57 |      0% |     1-129 |
| autoemulate/utils.py                                                                 |      159 |       12 |     92% |59, 67, 100, 184, 190, 228, 375-376, 403, 450, 459, 473 |
| tests/\_\_init\_\_.py                                                                |        0 |        0 |    100% |           |
| tests/experimental/conftest.py                                                       |       41 |        0 |    100% |           |
| tests/experimental/test\_device.py                                                   |       11 |        0 |    100% |           |
| tests/experimental/test\_experimental\_base.py                                       |       79 |        1 |     99% |        46 |
| tests/experimental/test\_experimental\_base\_simulator.py                            |       83 |        0 |    100% |           |
| tests/experimental/test\_experimental\_compare.py                                    |       28 |        0 |    100% |           |
| tests/experimental/test\_experimental\_conditional\_neural\_process.py               |       50 |        8 |     84% |     83-90 |
| tests/experimental/test\_experimental\_gaussian\_process\_exact.py                   |       59 |        0 |    100% |           |
| tests/experimental/test\_experimental\_lightgbm.py                                   |       32 |        0 |    100% |           |
| tests/experimental/test\_experimental\_model\_selection.py                           |       33 |        2 |     94% |    30, 34 |
| tests/experimental/test\_experimental\_preprocessors.py                              |       49 |        0 |    100% |           |
| tests/experimental/test\_experimental\_random\_forest.py                             |       37 |        0 |    100% |           |
| tests/experimental/test\_experimental\_svm.py                                        |       31 |        0 |    100% |           |
| tests/experimental/test\_experimental\_utils.py                                      |      218 |        0 |    100% |           |
| tests/experimental/test\_learners.py                                                 |       44 |        4 |     91% |     38-59 |
| tests/models/test\_attn\_cnp.py                                                      |      147 |        0 |    100% |           |
| tests/models/test\_cnp.py                                                            |      108 |        0 |    100% |           |
| tests/models/test\_cnp\_dataset.py                                                   |       74 |        0 |    100% |           |
| tests/models/test\_gptorch.py                                                        |       81 |        5 |     94% |     77-81 |
| tests/test\_base\_simulator.py                                                       |       97 |        2 |     98% |     45-46 |
| tests/test\_compare.py                                                               |      183 |        5 |     97% |239, 263, 272, 377-378 |
| tests/test\_cross\_validate.py                                                       |       97 |        3 |     97% | 56-59, 64 |
| tests/test\_data\_splitting.py                                                       |       11 |        0 |    100% |           |
| tests/test\_datasets.py                                                              |       13 |        0 |    100% |           |
| tests/test\_end\_to\_end.py                                                          |       39 |        0 |    100% |           |
| tests/test\_estimators.py                                                            |       30 |        0 |    100% |           |
| tests/test\_experimental\_design.py                                                  |       22 |        0 |    100% |           |
| tests/test\_gaussian\_process\_utils.py                                              |       76 |        0 |    100% |           |
| tests/test\_history\_matching.py                                                     |       58 |        0 |    100% |           |
| tests/test\_hyperparam\_searching.py                                                 |       48 |        0 |    100% |           |
| tests/test\_logging\_config.py                                                       |       51 |        0 |    100% |           |
| tests/test\_model\_processing.py                                                     |      100 |       17 |     83% |86-96, 106-119, 133, 157, 196, 221 |
| tests/test\_model\_registry.py                                                       |       86 |        0 |    100% |           |
| tests/test\_plotting.py                                                              |      205 |        7 |     97% |44, 54, 76-77, 85-86, 94 |
| tests/test\_preprocess\_target.py                                                    |      121 |        0 |    100% |           |
| tests/test\_printing.py                                                              |       19 |        0 |    100% |           |
| tests/test\_pytorch\_utils.py                                                        |       63 |        0 |    100% |           |
| tests/test\_save.py                                                                  |       62 |        2 |     97% |    30, 35 |
| tests/test\_sensitivity\_analysis.py                                                 |      116 |        0 |    100% |           |
| tests/test\_ui.py                                                                    |       57 |        0 |    100% |           |
| tests/test\_utils.py                                                                 |      202 |        6 |     97% |52, 58, 63, 68, 73, 78 |
|                                                                            **TOTAL** | **8270** | **1941** | **77%** |           |


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