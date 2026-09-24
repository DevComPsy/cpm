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

### Warnings about overflow in the Softmax

While the optimiser explores the parameter space, it can try very large inverse temperatures, where the exponentials in the Softmax overflow.
These warnings are usually harmless, because the optimiser moves away from those values.
If they persist, lower the upper bound of the inverse temperature, or rescale the values that go into the Softmax.

### A parameter estimate sits exactly at its bound

The data may carry little information about that parameter, or the bound may be too narrow.
Check whether the parameter can be recovered with a {doc}`parameter recovery study </tutorials/parameter-recovery>`, and consider fitting with priors (`prior=True`) or with {doc}`hierarchical estimation </tutorials/hierarchical-empirical-bayes>`.
