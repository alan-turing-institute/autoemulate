# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/alan-turing-institute/autoemulate/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                         |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| autoemulate/\_\_init\_\_.py                                                  |        0 |        0 |    100% |           |
| autoemulate/compare.py                                                       |      340 |       80 |     76% |267-277, 333-341, 430-440, 447, 451-459, 471, 488, 495-500, 506, 566, 582, 584, 587, 605, 609, 612, 626, 653, 658-660, 667, 672, 680, 690, 749, 755-756, 765-772, 785, 798-799, 823, 858, 864, 950-980, 1039-1060, 1084-1087 |
| autoemulate/cross\_validate.py                                               |       46 |        3 |     93% |81-82, 153 |
| autoemulate/data\_splitting.py                                               |        6 |        0 |    100% |           |
| autoemulate/datasets.py                                                      |       12 |        0 |    100% |           |
| autoemulate/emulators/\_\_init\_\_.py                                        |       22 |        0 |    100% |           |
| autoemulate/emulators/conditional\_neural\_process.py                        |      103 |        7 |     93% |254-255, 261-277, 282, 296 |
| autoemulate/emulators/conditional\_neural\_process\_attn.py                  |        9 |        0 |    100% |           |
| autoemulate/emulators/gaussian\_process.py                                   |      104 |        9 |     91% |73, 274, 282, 289, 300, 310, 321, 332, 340 |
| autoemulate/emulators/gaussian\_process\_mogp.py                             |       33 |       33 |      0% |      1-82 |
| autoemulate/emulators/gaussian\_process\_mt.py                               |       97 |        9 |     91% |71, 240, 246, 254, 265, 271, 278, 286, 290 |
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
| autoemulate/experimental/\_\_init\_\_.py                                     |        0 |        0 |    100% |           |
| autoemulate/experimental/calibration/\_\_init\_\_.py                         |        0 |        0 |    100% |           |
| autoemulate/experimental/calibration/bayes.py                                |      124 |       49 |     60% |87-88, 93-94, 100-102, 114-120, 131-149, 173-175, 181, 185-187, 241-249, 307-337 |
| autoemulate/experimental/calibration/history\_matching.py                    |      117 |       12 |     90% |65, 96-97, 103-104, 177, 253, 311, 316-317, 364, 389 |
| autoemulate/experimental/calibration/history\_matching\_dashboard.py         |      272 |      272 |      0% |     1-797 |
| autoemulate/experimental/callbacks/early\_stopping.py                        |       52 |        9 |     83% |69-71, 75, 102, 126, 131, 146, 151 |
| autoemulate/experimental/compare.py                                          |      200 |       83 |     58% |117, 122-126, 156, 174-177, 194-195, 205-209, 373-466, 488-492, 494-498, 512 |
| autoemulate/experimental/data/preprocessors.py                               |       25 |        0 |    100% |           |
| autoemulate/experimental/data/utils.py                                       |      187 |       21 |     89% |67, 100, 106, 122-126, 140, 156-158, 179, 184, 417, 420-422, 448, 451-453, 483 |
| autoemulate/experimental/datasets.py                                         |       14 |       14 |      0% |      1-67 |
| autoemulate/experimental/device.py                                           |       45 |        8 |     82% |15-16, 44, 99, 105, 153-158, 163 |
| autoemulate/experimental/emulators/\_\_init\_\_.py                           |       16 |        1 |     94% |        48 |
| autoemulate/experimental/emulators/base.py                                   |      188 |       16 |     91% |64, 68-69, 105-109, 200-203, 256-257, 369, 376, 455-457, 473 |
| autoemulate/experimental/emulators/ensemble.py                               |      107 |        8 |     93% |61, 70-71, 95-96, 203, 214-215 |
| autoemulate/experimental/emulators/gaussian\_process/\_\_init\_\_.py         |        8 |        0 |    100% |           |
| autoemulate/experimental/emulators/gaussian\_process/exact.py                |      155 |        3 |     98% |201, 208, 235 |
| autoemulate/experimental/emulators/gaussian\_process/kernel.py               |       36 |        0 |    100% |           |
| autoemulate/experimental/emulators/gaussian\_process/mean.py                 |       13 |        0 |    100% |           |
| autoemulate/experimental/emulators/gaussian\_process/poly\_mean.py           |       28 |        2 |     93% |    57, 68 |
| autoemulate/experimental/emulators/gaussian\_process/polynomial\_features.py |       35 |        2 |     94% |     72-73 |
| autoemulate/experimental/emulators/gradient\_boosting.py                     |       29 |        1 |     97% |        68 |
| autoemulate/experimental/emulators/lightgbm.py                               |       49 |        0 |    100% |           |
| autoemulate/experimental/emulators/nn/\_\_init\_\_.py                        |        0 |        0 |    100% |           |
| autoemulate/experimental/emulators/nn/mlp.py                                 |       38 |        0 |    100% |           |
| autoemulate/experimental/emulators/polynomials.py                            |       36 |        1 |     97% |        60 |
| autoemulate/experimental/emulators/radial\_basis\_functions.py               |       34 |        0 |    100% |           |
| autoemulate/experimental/emulators/random\_forest.py                         |       26 |        0 |    100% |           |
| autoemulate/experimental/emulators/svm.py                                    |       31 |        0 |    100% |           |
| autoemulate/experimental/emulators/transformed/\_\_init\_\_.py               |        0 |        0 |    100% |           |
| autoemulate/experimental/emulators/transformed/base.py                       |       89 |       14 |     84% |188-190, 324, 367-368, 395-398, 403-408, 412-416 |
| autoemulate/experimental/learners/\_\_init\_\_.py                            |        4 |        0 |    100% |           |
| autoemulate/experimental/learners/base.py                                    |      120 |       24 |     80% |69-72, 85-89, 102-111, 130, 134-141 |
| autoemulate/experimental/learners/membership.py                              |        8 |        8 |      0% |      1-19 |
| autoemulate/experimental/learners/pool.py                                    |        8 |        8 |      0% |      1-19 |
| autoemulate/experimental/learners/stream.py                                  |      131 |       13 |     90% |85-95, 141, 488, 539-543, 546, 574-579, 582 |
| autoemulate/experimental/logging\_config.py                                  |       58 |       21 |     64% |31, 34-38, 54-71, 99 |
| autoemulate/experimental/model\_selection.py                                 |       70 |        2 |     97% |   50, 138 |
| autoemulate/experimental/plotting.py                                         |       57 |        8 |     86% |28-29, 33-34, 84-85, 94, 166 |
| autoemulate/experimental/results.py                                          |       54 |        0 |    100% |           |
| autoemulate/experimental/save.py                                             |       73 |        7 |     90% |43-47, 73, 84-86 |
| autoemulate/experimental/sensitivity\_analysis.py                            |      247 |      146 |     41% |59-63, 74-75, 78-79, 81-82, 88-89, 112-113, 127-131, 155-159, 228-229, 239-240, 247, 275, 300, 328-332, 399-407, 412-415, 420-437, 465-504, 526-542, 566-667, 680-756 |
| autoemulate/experimental/simulations/\_\_init\_\_.py                         |        0 |        0 |    100% |           |
| autoemulate/experimental/simulations/base.py                                 |       66 |        2 |     97% |  103, 179 |
| autoemulate/experimental/simulations/epidemic.py                             |       40 |        0 |    100% |           |
| autoemulate/experimental/simulations/experimental\_design.py                 |       19 |        1 |     95% |        71 |
| autoemulate/experimental/simulations/projectile.py                           |       59 |        0 |    100% |           |
| autoemulate/experimental/simulations/reaction\_diffusion.py                  |       73 |       73 |      0% |     1-236 |
| autoemulate/experimental/transforms/\_\_init\_\_.py                          |        6 |        0 |    100% |           |
| autoemulate/experimental/transforms/base.py                                  |      130 |       31 |     76% |47-48, 68, 86, 89, 92-94, 138-141, 186-187, 219-220, 266, 305, 367-377, 381-382, 405-406, 446-447, 519-532 |
| autoemulate/experimental/transforms/pca.py                                   |       35 |        3 |     91% |     55-57 |
| autoemulate/experimental/transforms/standardize.py                           |       32 |        3 |     91% |     48-50 |
| autoemulate/experimental/transforms/utils.py                                 |       55 |        8 |     85% |63-64, 78-82, 128-136 |
| autoemulate/experimental/transforms/vae.py                                   |      127 |        9 |     93% |181, 209-213, 233-235, 258-262 |
| autoemulate/experimental/tuner.py                                            |       31 |        1 |     97% |        58 |
| autoemulate/experimental/types.py                                            |       19 |        0 |    100% |           |
| autoemulate/experimental\_design.py                                          |       19 |        3 |     84% |24, 35, 46 |
| autoemulate/history\_matching.py                                             |      115 |       41 |     64% |67, 71, 132-134, 206, 210-217, 230, 280, 293-295, 304-306, 326, 334-357, 404-432 |
| autoemulate/hyperparam\_searching.py                                         |       46 |        3 |     93% |93, 99-100 |
| autoemulate/logging\_config.py                                               |       43 |        4 |     91% |28, 56, 63-64 |
| autoemulate/mcmc.py                                                          |      160 |      160 |      0% |     1-348 |
| autoemulate/mcmc\_dashboard.py                                               |      369 |      369 |      0% |     1-816 |
| autoemulate/metrics.py                                                       |        7 |        0 |    100% |           |
| autoemulate/model\_processing.py                                             |       48 |        1 |     98% |        67 |
| autoemulate/model\_registry.py                                               |       31 |        1 |     97% |        46 |
| autoemulate/plotting.py                                                      |      179 |       10 |     94% |52, 145, 155, 249, 423, 428, 438, 498, 618-619 |
| autoemulate/preprocess\_target.py                                            |      263 |       28 |     89% |87, 99, 119, 179, 211-212, 219, 259, 284, 322, 455-477, 492, 497, 527, 530, 533, 573, 614 |
| autoemulate/printing.py                                                      |       41 |       14 |     66% |7, 12, 17-26, 40, 128, 137-139 |
| autoemulate/save.py                                                          |       36 |        3 |     92% |     28-30 |
| autoemulate/sensitivity\_analysis.py                                         |      213 |      135 |     37% |55-77, 85, 88, 90, 95, 123, 167-170, 269-283, 306-344, 376-399, 420-436, 455-555, 564-636 |
| autoemulate/simulations/\_\_init\_\_.py                                      |        0 |        0 |    100% |           |
| autoemulate/simulations/base.py                                              |       81 |       16 |     80% |60, 78, 103, 114, 136-139, 157, 172-188, 211-217 |
| autoemulate/simulations/circ\_utils.py                                       |       94 |       94 |      0% |     4-233 |
| autoemulate/simulations/epidemic.py                                          |       35 |        0 |    100% |           |
| autoemulate/simulations/flow\_functions.py                                   |       85 |       85 |      0% |     1-162 |
| autoemulate/simulations/naghavi\_cardiac\_ModularCirc.py                     |       70 |       70 |      0% |     1-148 |
| autoemulate/simulations/projectile.py                                        |       55 |        8 |     85% |207-212, 229-230, 251-253 |
| autoemulate/utils.py                                                         |      159 |       12 |     92% |59, 67, 100, 184, 190, 228, 375-376, 403, 450, 459, 473 |
| tests/\_\_init\_\_.py                                                        |        0 |        0 |    100% |           |
| tests/experimental/\_\_init\_\_.py                                           |        0 |        0 |    100% |           |
| tests/experimental/conftest.py                                               |       73 |        0 |    100% |           |
| tests/experimental/test\_device.py                                           |       11 |        0 |    100% |           |
| tests/experimental/test\_experimental\_base.py                               |      116 |        1 |     99% |        51 |
| tests/experimental/test\_experimental\_base\_simulator.py                    |       83 |        0 |    100% |           |
| tests/experimental/test\_experimental\_bayesian\_calibration.py              |       55 |        0 |    100% |           |
| tests/experimental/test\_experimental\_compare.py                            |       32 |        0 |    100% |           |
| tests/experimental/test\_experimental\_conditional\_neural\_process.py       |        0 |        0 |    100% |           |
| tests/experimental/test\_experimental\_early\_stopping.py                    |       27 |        0 |    100% |           |
| tests/experimental/test\_experimental\_ensemble.py                           |       70 |        0 |    100% |           |
| tests/experimental/test\_experimental\_gaussian\_process\_exact.py           |      106 |        0 |    100% |           |
| tests/experimental/test\_experimental\_gradient\_boosting.py                 |       20 |        0 |    100% |           |
| tests/experimental/test\_experimental\_history\_matching.py                  |       73 |        2 |     97% |     23-24 |
| tests/experimental/test\_experimental\_lightgbm.py                           |       43 |        0 |    100% |           |
| tests/experimental/test\_experimental\_mlp.py                                |       56 |        0 |    100% |           |
| tests/experimental/test\_experimental\_model\_selection.py                   |       40 |        4 |     90% |31-32, 37, 41 |
| tests/experimental/test\_experimental\_plotting.py                           |       28 |        0 |    100% |           |
| tests/experimental/test\_experimental\_polynomials.py                        |       55 |        0 |    100% |           |
| tests/experimental/test\_experimental\_preprocessors.py                      |       49 |        0 |    100% |           |
| tests/experimental/test\_experimental\_radial\_basis\_functions.py           |       49 |        0 |    100% |           |
| tests/experimental/test\_experimental\_random\_forest.py                     |       48 |        0 |    100% |           |
| tests/experimental/test\_experimental\_save.py                               |      116 |        3 |     97% | 34-35, 40 |
| tests/experimental/test\_experimental\_sensitivity\_analysis.py              |       89 |        0 |    100% |           |
| tests/experimental/test\_experimental\_svm.py                                |       40 |        0 |    100% |           |
| tests/experimental/test\_experimental\_transformed.py                        |      108 |        0 |    100% |           |
| tests/experimental/test\_experimental\_utils.py                              |      218 |        0 |    100% |           |
| tests/experimental/test\_learners.py                                         |       41 |        4 |     90% |     33-54 |
| tests/experimental/test\_results.py                                          |       74 |        0 |    100% |           |
| tests/experimental/test\_utils.py                                            |       11 |        0 |    100% |           |
| tests/experimental/transforms/test\_serde.py                                 |       86 |        1 |     99% |       146 |
| tests/experimental/transforms/test\_transforms.py                            |       70 |        0 |    100% |           |
| tests/models/test\_attn\_cnp.py                                              |      147 |        0 |    100% |           |
| tests/models/test\_cnp.py                                                    |      108 |        0 |    100% |           |
| tests/models/test\_cnp\_dataset.py                                           |       74 |        0 |    100% |           |
| tests/models/test\_gptorch.py                                                |       81 |        5 |     94% |     77-81 |
| tests/test\_base\_simulator.py                                               |       97 |        2 |     98% |     45-46 |
| tests/test\_compare.py                                                       |      183 |        5 |     97% |239, 263, 272, 377-378 |
| tests/test\_cross\_validate.py                                               |       97 |        3 |     97% | 56-59, 64 |
| tests/test\_data\_splitting.py                                               |       11 |        0 |    100% |           |
| tests/test\_datasets.py                                                      |       13 |        0 |    100% |           |
| tests/test\_end\_to\_end.py                                                  |       39 |        0 |    100% |           |
| tests/test\_estimators.py                                                    |       29 |        0 |    100% |           |
| tests/test\_experimental\_design.py                                          |       22 |        0 |    100% |           |
| tests/test\_gaussian\_process\_utils.py                                      |       76 |        0 |    100% |           |
| tests/test\_history\_matching.py                                             |       58 |        0 |    100% |           |
| tests/test\_hyperparam\_searching.py                                         |       48 |        0 |    100% |           |
| tests/test\_logging\_config.py                                               |       51 |        0 |    100% |           |
| tests/test\_model\_processing.py                                             |      100 |       17 |     83% |86-96, 106-119, 133, 157, 196, 221 |
| tests/test\_model\_registry.py                                               |       86 |        0 |    100% |           |
| tests/test\_plotting.py                                                      |      205 |        7 |     97% |44, 54, 76-77, 85-86, 94 |
| tests/test\_preprocess\_target.py                                            |      121 |        0 |    100% |           |
| tests/test\_printing.py                                                      |       19 |        0 |    100% |           |
| tests/test\_pytorch\_utils.py                                                |       63 |        0 |    100% |           |
| tests/test\_save.py                                                          |       62 |        2 |     97% |    30, 35 |
| tests/test\_sensitivity\_analysis.py                                         |      116 |        0 |    100% |           |
| tests/test\_ui.py                                                            |       57 |        0 |    100% |           |
| tests/test\_utils.py                                                         |      202 |        6 |     97% |52, 58, 63, 68, 73, 78 |
|                                                                    **TOTAL** | **11040** | **2150** | **81%** |           |


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