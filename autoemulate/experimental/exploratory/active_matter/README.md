# Exploring ["The Well" Active matter simulations](https://polymathic-ai.org/the_well/datasets/active_matter/#active-matter-simulations) dataset

This notebook explores the dataset and roughly follows the [dataset tutorial](https://polymathic-ai.org/the_well/tutorials/dataset/) found on The Well's documentation, but looks specifically at the active matter simulations dataset, with clarifying notes. This is a prelude to using the dataset as an example for The Alan Turing Institute's [AutoEmulate](https://github.com/alan-turing-institute/autoemulate) project.

### Dataset notes (see [paper](https://arxiv.org/abs/2308.06675))

Active matter systems consist of collections of agents, such as particles or macromolecules, that convert chemical energy into mechanical work; A few prominent experimental realizations include suspensions of swimming bacteria and in vitro mixtures of cytoskeletal filaments and motor proteins.

### Notebook requirements

```
matplotlib
numpy
torch
the_well
ipykernel
tdqm
huggingface-hub
```

Save these to a requirements.txt file in the same directory as the notebook.
Then run the following command to install the requirements:

```
pip install -r requirements.txt
```