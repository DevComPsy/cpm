# Installation

cpm is written in Python, and needs Python 3.11 or newer.
Its dependencies, such as NumPy, SciPy and pandas, are installed with it.

We recommend installing cpm in a virtual environment, so that its dependencies do not interfere with other projects:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

## Install cpm

::::{tab-set}

:::{tab-item} From PyPI
The latest release:

```bash
pip install cpm-toolbox
```
:::

:::{tab-item} Development version
The latest version from GitHub, with changes that are not released yet:

```bash
pip install git+https://github.com/DevComPsy/cpm.git
```
:::

:::{tab-item} From source
To work on cpm itself, clone the repository and install it in editable mode:

```bash
git clone https://github.com/DevComPsy/cpm.git
cd cpm
pip install -e .
```

See {doc}`/about/contributing` for the development setup.
:::

::::

The package is called `cpm-toolbox` on PyPI, and `cpm` in Python.

## Check the installation

```python
import cpm

print(cpm.__version__)
```

## Extras for the tutorials

The tutorials and examples also use [matplotlib](https://matplotlib.org/) for figures, which is installed with cpm.
To run all examples, install the optional notebook dependencies too:

```bash
pip install "cpm-toolbox[notebooks]"
```

## Problems?

Some dependencies, such as SciPy, may need a compiler on some systems if no prebuilt package is available for your platform and Python version.
If the installation fails, see {doc}`troubleshooting`.
