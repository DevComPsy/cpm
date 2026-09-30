# Speed up fitting with numba

Fitting a model evaluates it thousands of times per participant. A {py:class}`~cpm.generators.Wrapper` calls the model function once per trial, and that call, not the model itself, is what most of the time goes into. The built-in applications in {py:mod}`cpm.applications` therefore compute all trials of a participant at once, and with [numba](https://numba.pydata.org/) installed, that loop over trials is compiled as a whole. You can do the same for your own models with {py:class}`~cpm.generators.SessionWrapper`.

## Install numba

numba is an optional dependency. Install it with cpm (see {doc}`/get-started/installation` for the other ways to install it):

```bash
pip install "cpm-toolbox[numba]"
```

That is all: the applications use numba whenever it can be imported, and your code stays the same. Without numba, they run the same computation as plain Python, with the same results. numba supports a new NumPy release some time after it comes out, and does not support PyPy; if numba is not installed, cpm uses plain Python without further notice, and if it is installed but cannot be imported, cpm warns once, with numba's error, and uses plain Python.

## What to expect

One evaluation of the objective function, with the parameters' priors, for one participant (from `python benchmarks/run.py` in the cpm repository):

| Application | Trials | Without numba | With numba | Per trial, in a `Wrapper` |
|---|---:|---:|---:|---:|
| `RLRW` | 71 | 0.66 ms | 0.08 ms | 3.7 ms |
| `HybridMBMF` | 200 | 1.8 ms | 0.11 ms | 11 ms |
| `PTSM` | 40 | 0.26 ms | 0.07 ms | 2.1 ms |
| `PTSM1992` | 40 | 0.31 ms | 0.08 ms | 2.1 ms |
| `PTSM2025` | 40 | 0.31 ms | 0.07 ms | 1.8 ms |

With numba, the model itself takes 0.02 to 0.05 ms of this, and the loss function about 0.02 ms. The last column is the same model computed one trial at a time, see [below](#extend-an-application-trial-by-trial).

The first time an application runs with numba, numba compiles it, which takes a few seconds: about 3 to 5 s for the reinforcement-learning models, and up to about 15 s for the prospect-theory models. The machine code is cached on disk, next to cpm's own files, so later runs, new Python sessions and the worker processes of a parallel fit (see {doc}`parallelise-fitting`) load it in a fraction of a second. If cpm is installed in a directory you cannot write to, numba caches in your user directory instead; if it cannot write there either, as on some clusters, it compiles the models again in every new Python session.

To run without numba although it is installed, for example to compare, set the environment variable `CPM_DISABLE_JIT=1` before starting Python.

## Simulations and random numbers

Where a model samples choices, as with `generate=True`, the applications draw one uniform random number per trial from `numpy.random` and turn it into a choice exactly as `numpy.random.choice` does. A simulation after `numpy.random.seed(...)` therefore makes the same choices with and without numba.

## Extend an application trial by trial

The `model` attribute of an application is still its model function for a single trial, `model(parameters, trial)`, which computes one trial from the states in `parameters` and returns that trial's outputs, including the updated states. You can call it from your own per-trial model, for example to give the parameters of an application different values on different kinds of trials, and wrap that in a {py:class}`~cpm.generators.Wrapper`:

```python
from cpm.applications.reinforcement_learning import HybridMBMF
from cpm.generators import Wrapper

application = HybridMBMF(data=two_step_data_of_one_participant)
hybrid = application.model

def model(parameters, trial):
    ## ... for example, change the parameters for this trial ...
    return hybrid(parameters=parameters, trial=trial)

wrapper = Wrapper(model=model, data=two_step_data_of_one_participant, parameters=application.parameters)
wrapper.run()
```

This gives the same results as the application, one trial at a time, and runs at the speed of a per-trial `Wrapper` (the last column above). To get the speed of the application back, write the extended model for all trials at once, as below.

## Write your own session model

A session model is a function of `parameters` and `data` that computes all trials at once and returns one row per trial. Wrap it in a {py:class}`~cpm.generators.SessionWrapper`:

```python
import numpy as np
from cpm.generators import SessionWrapper, Parameters, Value

def model(parameters, data):
    alpha = parameters.alpha.value
    beta = parameters.beta.value
    values = np.asarray(parameters.values, dtype=float).copy()
    n = len(data["choice"])
    p_right = np.empty(n)
    for t in range(n):
        p_right[t] = 1 / (1 + np.exp(-beta * (values[1] - values[0])))
        choice = data["choice"][t]
        values[choice] += alpha * (data["reward"][t] - values[choice])
    return {"p_right": p_right, "dependent": p_right}

parameters = Parameters(
    alpha=Value(value=0.3, lower=0, upper=1, prior="truncated_normal", args={"mean": 0.5, "sd": 0.25}),
    beta=Value(value=2.0, lower=0, upper=10, prior="truncated_normal", args={"mean": 5, "sd": 2.5}),
    values=np.array([0.5, 0.5]),
)
wrapper = SessionWrapper(model=model, data=data_of_one_participant, parameters=parameters)
```

`data` arrives as a dictionary of numpy arrays, one per column, converted once when the data are set (pass `prepare` to convert them differently). Outputs with one value per trial become one column of `export()`, and outputs with several values per trial become several columns, as with a `Wrapper`.

This alone, in plain Python, is typically 5 to 15 times faster than the same model in a per-trial `Wrapper`. To compile it, move the loop into a function of numbers and arrays only, decorate it with `numba.njit`, and call it from the model function:

```python
from numba import njit
from cpm.models import kernels

@njit
def learn(alpha, beta, values, choices, rewards):
    values = values.copy()
    p_right = np.empty(choices.shape[0])
    for t in range(choices.shape[0]):
        p_right[t] = kernels.p_second(beta, values[0], values[1])
        values[choices[t]] += alpha * (rewards[t] - values[choices[t]])
    return p_right

def model(parameters, data):
    p_right = learn(
        float(parameters.alpha), float(parameters.beta),
        np.asarray(parameters.values, dtype=float), data["choice"], data["reward"],
    )
    return {"p_right": p_right, "dependent": p_right}
```

`@njit(cache=True)` keeps the compiled code on disk between Python sessions. This works in scripts, modules and Jupyter notebooks; at the plain Python prompt, numba cannot cache and raises an error, so leave out `cache=True` there.
{py:mod}`cpm.models.kernels` holds the formulas of the learning rules, decision rules, activation functions and attention mechanisms in {py:mod}`cpm.models` as functions: the classes compute with them, and compiled loops like this one can call them directly.

Inside a compiled function, use numbers, numpy arrays, and functions from `math`, `numpy` and {py:mod}`cpm.models.kernels`. pandas objects, dictionaries of mixed types, cpm's `Parameters` and `Value` objects, and most other Python objects are not available there, which is why the model function unpacks them first. See numba's [list of supported Python and NumPy features](https://numba.readthedocs.io/en/stable/reference/pysupported.html).

## Debug a compiled model

Set the environment variable `NUMBA_DISABLE_JIT=1` before starting Python to run all compiled functions, cpm's and your own, as plain Python, where you can use a debugger and `print` and get ordinary tracebacks. cpm's own `CPM_DISABLE_JIT=1` does not do this: it turns off only cpm's compilation, so the functions in {py:mod}`cpm.models.kernels` become plain Python functions, and a compiled function of yours that calls them can no longer be compiled. numba's error messages for code it cannot compile name the line and the types involved; the most common cause is passing a pandas object, a list or a `Value` into a compiled function.
