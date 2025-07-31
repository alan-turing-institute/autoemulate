# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                            |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| autoemulate/\_\_init\_\_.py                                     |        0 |        0 |    100% |           |
| autoemulate/calibration/\_\_init\_\_.py                         |        0 |        0 |    100% |           |
| autoemulate/calibration/bayes.py                                |      124 |       49 |     60% |89-90, 95-96, 102-104, 116-122, 131-149, 172-174, 180, 184-186, 239-247, 305-335 |
| autoemulate/calibration/history\_matching.py                    |      116 |       12 |     90% |67, 98-99, 105-106, 181, 257, 317, 322-323, 372, 397 |
| autoemulate/calibration/history\_matching\_dashboard.py         |      272 |      272 |      0% |     1-798 |
| autoemulate/callbacks/early\_stopping.py                        |       52 |        9 |     83% |70-72, 77, 103, 127, 132, 147, 152 |
| autoemulate/core/\_\_init\_\_.py                                |        0 |        0 |    100% |           |
| autoemulate/core/compare.py                                     |      210 |       87 |     59% |125, 130-134, 164, 183-186, 204-205, 216-220, 380-389, 427-520, 542-546, 548-552, 566 |
| autoemulate/core/datasets.py                                    |       14 |       14 |      0% |      1-67 |
| autoemulate/core/device.py                                      |       45 |        8 |     82% |15-16, 44, 99, 105, 153-158, 163 |
| autoemulate/core/logging\_config.py                             |       58 |       21 |     64% |30, 33-37, 53-70, 114 |
| autoemulate/core/model\_selection.py                            |       70 |        2 |     97% |   53, 140 |
| autoemulate/core/plotting.py                                    |       57 |        8 |     86% |30-31, 35-36, 85-86, 95, 167 |
| autoemulate/core/results.py                                     |       54 |        0 |    100% |           |
| autoemulate/core/save.py                                        |       73 |        7 |     90% |45-49, 75, 86-88 |
| autoemulate/core/sensitivity\_analysis.py                       |      247 |      146 |     41% |59-63, 72-73, 76-77, 79-80, 86-87, 110-111, 125-129, 152-156, 223-224, 234-235, 242, 271, 296, 324-328, 395-403, 408-411, 416-433, 461-500, 522-538, 562-663, 675-751 |
| autoemulate/core/tuner.py                                       |       43 |        4 |     91% |60, 151-158 |
| autoemulate/core/types.py                                       |       19 |        0 |    100% |           |
| autoemulate/data/utils.py                                       |      187 |       21 |     89% |61, 92, 98, 114-118, 130, 146-148, 169, 174, 405, 408-410, 438, 441-443, 475 |
| autoemulate/emulators/\_\_init\_\_.py                           |       18 |        1 |     94% |        49 |
| autoemulate/emulators/base.py                                   |      189 |       19 |     90% |87, 105-106, 115, 121-126, 161-165, 259-262, 338-339, 465, 472, 553-555, 571 |
| autoemulate/emulators/ensemble.py                               |      112 |        9 |     92% |63, 68, 77-78, 102-103, 247, 258-259 |
| autoemulate/emulators/gaussian\_process/\_\_init\_\_.py         |        8 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process/exact.py                |      156 |        3 |     98% |200, 207, 234 |
| autoemulate/emulators/gaussian\_process/kernel.py               |       36 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process/mean.py                 |       13 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process/poly\_mean.py           |       28 |        2 |     93% |    56, 69 |
| autoemulate/emulators/gaussian\_process/polynomial\_features.py |       35 |        2 |     94% |     73-74 |
| autoemulate/emulators/gradient\_boosting.py                     |       32 |        1 |     97% |       114 |
| autoemulate/emulators/lightgbm.py                               |       52 |        0 |    100% |           |
| autoemulate/emulators/nn/\_\_init\_\_.py                        |        0 |        0 |    100% |           |
| autoemulate/emulators/nn/mlp.py                                 |       41 |        0 |    100% |           |
| autoemulate/emulators/polynomials.py                            |       39 |        1 |     97% |        95 |
| autoemulate/emulators/radial\_basis\_functions.py               |       36 |        0 |    100% |           |
| autoemulate/emulators/random\_forest.py                         |       30 |        0 |    100% |           |
| autoemulate/emulators/svm.py                                    |       35 |        0 |    100% |           |
| autoemulate/emulators/transformed/\_\_init\_\_.py               |        0 |        0 |    100% |           |
| autoemulate/emulators/transformed/base.py                       |       89 |       14 |     84% |187-189, 324, 367-368, 395-398, 403-408, 417-421 |
| autoemulate/learners/\_\_init\_\_.py                            |        4 |        0 |    100% |           |
| autoemulate/learners/base.py                                    |      121 |       26 |     79% |69-72, 87-91, 106-115, 142, 146-153, 243-244 |
| autoemulate/learners/membership.py                              |        8 |        8 |      0% |      1-19 |
| autoemulate/learners/pool.py                                    |        8 |        8 |      0% |      1-19 |
| autoemulate/learners/stream.py                                  |      131 |       13 |     90% |85-95, 141, 492, 543-547, 550, 578-583, 586 |
| autoemulate/simulations/\_\_init\_\_.py                         |        6 |        0 |    100% |           |
| autoemulate/simulations/base.py                                 |       92 |        4 |     96% |59, 153, 224-228 |
| autoemulate/simulations/epidemic.py                             |       40 |        0 |    100% |           |
| autoemulate/simulations/experimental\_design.py                 |       32 |        0 |    100% |           |
| autoemulate/simulations/flow\_problem.py                        |       45 |       36 |     20% |50-75, 102-161 |
| autoemulate/simulations/projectile.py                           |       59 |        0 |    100% |           |
| autoemulate/simulations/reaction\_diffusion.py                  |       73 |       73 |      0% |     1-235 |
| autoemulate/transforms/\_\_init\_\_.py                          |        6 |        0 |    100% |           |
| autoemulate/transforms/base.py                                  |      130 |       31 |     76% |50-51, 71, 89, 92, 95-97, 141-144, 191-192, 226-227, 275, 316, 380-390, 394-395, 418-419, 461-462, 536-549 |
| autoemulate/transforms/pca.py                                   |       35 |        3 |     91% |     57-61 |
| autoemulate/transforms/standardize.py                           |       32 |        3 |     91% |     50-52 |
| autoemulate/transforms/utils.py                                 |       55 |        8 |     85% |64-65, 79-83, 129-137 |
| autoemulate/transforms/vae.py                                   |      127 |        9 |     93% |180, 208-212, 233-235, 258-262 |
| tests/\_\_init\_\_.py                                           |        0 |        0 |    100% |           |
| tests/conftest.py                                               |       73 |        0 |    100% |           |
| tests/test\_base.py                                             |      105 |        1 |     99% |        47 |
| tests/test\_base\_simulator.py                                  |      103 |        0 |    100% |           |
| tests/test\_bayesian\_calibration.py                            |       58 |        0 |    100% |           |
| tests/test\_compare.py                                          |       32 |        0 |    100% |           |
| tests/test\_conditional\_neural\_process.py                     |        0 |        0 |    100% |           |
| tests/test\_design.py                                           |       28 |        0 |    100% |           |
| tests/test\_device.py                                           |       11 |        0 |    100% |           |
| tests/test\_early\_stopping.py                                  |       27 |        0 |    100% |           |
| tests/test\_ensemble.py                                         |       70 |        0 |    100% |           |
| tests/test\_experimental\_utils.py                              |      218 |        0 |    100% |           |
| tests/test\_gaussian\_process\_exact.py                         |      103 |        0 |    100% |           |
| tests/test\_gradient\_boosting.py                               |       20 |        0 |    100% |           |
| tests/test\_history\_matching.py                                |       75 |        2 |     97% |     23-24 |
| tests/test\_learners.py                                         |       43 |        4 |     91% |     35-56 |
| tests/test\_lightgbm.py                                         |       43 |        0 |    100% |           |
| tests/test\_mlp.py                                              |       56 |        0 |    100% |           |
| tests/test\_model\_selection.py                                 |       40 |        4 |     90% |31-32, 37, 41 |
| tests/test\_plotting.py                                         |       28 |        0 |    100% |           |
| tests/test\_polynomials.py                                      |       55 |        0 |    100% |           |
| tests/test\_radial\_basis\_functions.py                         |       49 |        0 |    100% |           |
| tests/test\_random\_forest.py                                   |       48 |        0 |    100% |           |
| tests/test\_results.py                                          |       74 |        0 |    100% |           |
| tests/test\_save.py                                             |      116 |        3 |     97% | 34-35, 40 |
| tests/test\_sensitivity\_analysis.py                            |       89 |        0 |    100% |           |
| tests/test\_svm.py                                              |       40 |        0 |    100% |           |
| tests/test\_transformed.py                                      |      109 |        0 |    100% |           |
| tests/test\_utils.py                                            |       11 |        0 |    100% |           |
| tests/transforms/test\_serde.py                                 |       86 |        1 |     99% |       146 |
| tests/transforms/test\_transforms.py                            |       70 |        0 |    100% |           |
|                                                       **TOTAL** | **5474** |  **949** | **83%** |           |


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