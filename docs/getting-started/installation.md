# Installation instructions

`AutoEmulate` is a Python package that can be installed in a number of ways.
In this section we will describe the main ways to install the package.
For new users, we recommend installing the package from PyPI.
For users who want to contribute to the package, we recommend using Poetry to install the package from the source code.

## Prerequisites

**Python Version:** `AutoEmulate` requires Python `>=3.10` and `<3.13`.

## Install from PyPI

To get the latest release from PyPI:

```bash
pip install autoemulate
```

## Install using Poetry

If you'd like to contribute to `AutoEmulate`, you can install the package using Poetry.

* Ensure you have poetry installed. If not, install it following the [official instructions](https://python-poetry.org/docs/). This has been most recently tested with Poetry version `2.1`.

* Fork the repository on GitHub by clicking the "Fork" button at the top right of the [AutoEmulate repository](https://github.com/alan-turing-institute/autoemulate)

* Clone your forked repository:

```bash
git clone https://github.com/YOUR-USERNAME/autoemulate.git
```

Navigate into the directory:

```bash
cd autoemulate
```

Set up poetry:

```bash
poetry install
```

Create a virtual environment:

```bash
poetry env activate
```

Then activate the virtual environment using the command displayed by the previous command. This will be something like:

```bash
source /Users/yourName/Library/Caches/pypoetry/virtualenvs/autoemulate-l4vGdsmY-py3.11/bin/activate
```

## Interactive tutorials

You can run the Quickstart demo and other interactive tutorials fron the documentation locally.
The examples are all Jupyter notebooks and can be run in your favoured method, such as JupyterLab, Jupyter Notebook, or VS Code.

<details>
<summary>
These steps will guide you in the simplest way to set up a virtual environment, install the package from PyPI and run the notebooks with JupyterLab.
</summary>

1. Clone the AutoEmulate repository:

   ```bash
   git clone https://github.com/alan-turing-institute/autoemulate
   ```
2. Navigate into the directory:

   ```bash
   cd autoemulate
   ```
3. Set up a virtual environment called `autoemulate`:

   ```bash
   python -m venv autoemulate
   ```
4. Activate the virtual environment:
   - On Windows:

     ```bash
     autoemulate\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source autoemulate/bin/activate
     ```
5. Install the package from PyPI:

   ```bash
   pip install autoemulate
   ```
6. Install JupyterLab:

   ```bash
   pip install jupyterlab
   ```
7. Create a Jupyter kernel for the virtual environment:

   ```bash
   python -m ipykernel install --user --name autoemulate --display-name "Python (autoemulate)"
   ```

   This command registers the virtual environment as a Jupyter kernel named `Python (autoemulate)`, which you can select in JupyterLab.
8. Launch JupyterLab:

   ```bash
   jupyter lab
   ```
9. Open the `docs/getting-started/quickstart.ipynb` notebook in JupyterLab.
10. Set the kernel to use the `Python (autoemulate)` kernel you created earlier. You can do this by clicking on the kernel name in the top right corner of the JupyterLab interface and selecting `Python (autoemulate)` from the dropdown menu.
11. Find other interactive tutorials in the `docs/tutorials` directory, which you can open and run in JupyterLab.

</details>