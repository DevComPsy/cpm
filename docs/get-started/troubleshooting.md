# Troubleshooting

## Getting help

- **Questions** about how to do something with cpm: search the [GitHub Discussions](https://github.com/DevComPsy/cpm/discussions), and ask there if your question has not been answered yet.
- **Bugs**, and mistakes in the documentation: report them on the [GitHub issue tracker](https://github.com/DevComPsy/cpm/issues). Include the version of cpm (`cpm.__version__`), a short piece of code that reproduces the problem, and the full error message.

## Common problems

### `AttributeError: 'DataFrame' object has no attribute 'observed'`

The optimisers fit the model to a column called `observed`, which holds the behaviour the model should predict.
Add it to your data before fitting, for example `data["observed"] = data["response"]`.
See {doc}`concepts/data-format`.

### `TypeError: Data should be a pandas.DataFrameGroupBy object`

A {py:class}`~cpm.generators.Simulator` runs a model for many participants, so it takes the data grouped by participant: `data.groupby("ppt")`, not `data`.

### Parallel fitting starts, then hangs or restarts, on Windows or macOS

The code that runs the fit must be under an `if __name__ == "__main__":` guard.
See {doc}`/how-to/parallelise-fitting`.

### Warnings about overflow

While the optimiser explores the parameter space, it can try very large parameter values, such as inverse temperatures, where exponentials overflow.
{py:class}`~cpm.models.decision.Softmax` and the built-in models compute their probabilities in a way that cannot overflow, but other functions, such as {py:class}`~cpm.models.decision.Sigmoid` or the exponentials in your own model, can still warn about it.
These warnings are usually harmless, because the optimiser moves away from those values.
If they persist, lower the upper bound of the parameter, or rescale the values that go into the exponential.

### numba does not install, or does not work with my version of NumPy

numba is optional (see [Faster fitting with numba](installation.md#faster-fitting-with-numba)), and cpm works without it.
numba supports a new NumPy release some time after it comes out, so the newest NumPy may not have a numba release yet; `pip` then either installs an older NumPy for numba, or cannot install numba.
If numba is installed but cannot be imported, cpm runs the models as plain Python, with the same results, only more slowly.
To use numba, install a NumPy version that numba supports (see the [numba installation notes](https://numba.readthedocs.io/en/stable/user/installing.html)), for example in a separate virtual environment.

### The first fit of a built-in model takes a few seconds longer

With numba installed, the first run of each built-in model compiles it.
The compiled code is cached on disk, so this happens once per model and cpm version, not on every run.

### A parameter estimate sits exactly at its bound

The data may carry little information about that parameter, or the bound may be too narrow.
Check whether the parameter can be recovered with a {doc}`parameter recovery study </tutorials/parameter-recovery>`, and consider fitting with priors (`prior=True`) or with {doc}`hierarchical estimation </tutorials/hierarchical-empirical-bayes>`.
