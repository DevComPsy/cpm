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

With numba, for faster fitting (see [below](#faster-fitting-with-numba)):

```bash
pip install "cpm-toolbox[numba]"
```
:::

:::{tab-item} Development version
The latest version from GitHub, with changes that are not released yet:

```bash
pip install git+https://github.com/DevComPsy/cpm.git
```

With numba:

```bash
pip install "cpm-toolbox[numba] @ git+https://github.com/DevComPsy/cpm.git"
```
:::

:::{tab-item} From source
To work on cpm itself, clone the repository and install it in editable mode:

```bash
git clone https://github.com/DevComPsy/cpm.git
cd cpm
pip install -e .
```

With numba: `pip install -e ".[numba]"`.

See {doc}`/about/contributing` for the development setup.
:::

::::

The package is called `cpm-toolbox` on PyPI, and `cpm` in Python.

## Check the installation

```python
import cpm

print(cpm.__version__)
```

## Faster fitting with numba

[numba](https://numba.pydata.org/) is an optional dependency that compiles the built-in models ({py:class}`~cpm.applications.reinforcement_learning.RLRW`, {py:class}`~cpm.applications.reinforcement_learning.HybridMBMF`, {py:class}`~cpm.applications.decision_making.PTSM`, {py:class}`~cpm.applications.decision_making.PTSM1992` and {py:class}`~cpm.applications.decision_making.PTSM2025`), which makes fitting and simulating them many times faster.
Install it together with cpm:

```bash
pip install "cpm-toolbox[numba]"
```

or add it to an existing installation with the same command, or with `pip install "numba>=0.64"`.

Nothing else changes: your code stays the same, and cpm uses numba whenever it can import it. Without numba, cpm runs the same models as plain Python, with the same results, only more slowly. numba matters most when a model is run many times, as in fitting many participants, hierarchical estimation or parameter recovery; for a single run, it makes little difference.

- **Check whether numba is used.** cpm uses numba if `python -c "import numba"` runs without an error in the environment you run cpm in.
- **Compilation.** The first time a built-in model runs, numba compiles it, which takes a few seconds (up to about 15 s). The result is cached on disk, so later runs, new Python sessions and parallel fits start at full speed.
- **Limitations.** numba supports a new NumPy release some time after it comes out, and does not support PyPy. If numba cannot be installed or imported in your environment, cpm works as usual, without it.
- **Turning it off.** To run without numba although it is installed, for example to compare, set the environment variable `CPM_DISABLE_JIT=1` before starting Python.

See {doc}`/how-to/fast-session-models` for how much faster the models are, and how to compile your own models.

## Extras for the tutorials

The tutorials and examples also use [matplotlib](https://matplotlib.org/) for figures, which is installed with cpm.
To run all examples, install the optional notebook dependencies too:

```bash
pip install "cpm-toolbox[notebooks]"
```

## Problems?

Some dependencies, such as SciPy, may need a compiler on some systems if no prebuilt package is available for your platform and Python version.
If the installation fails, see {doc}`troubleshooting`.
