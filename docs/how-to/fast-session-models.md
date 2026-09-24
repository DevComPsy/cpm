# Speed up fitting with session models

Fitting a model evaluates it thousands of times per participant. A {py:class}`~cpm.generators.Wrapper` calls the model function once per trial, and that call, not the model itself, is what most of the time goes into. A {py:class}`~cpm.generators.SessionWrapper` calls the model function once for all trials of a participant instead, and with [numba](https://numba.pydata.org/) the loop over trials can be compiled as a whole. For the built-in applications, this makes one evaluation of the objective function 35 to 120 times faster, and a fit correspondingly faster.

## Use the session version of a built-in application

Every application in {py:mod}`cpm.applications` that is a `Wrapper` has a session version, named after it with `Session` appended:

| Per-trial | Session version |
|---|---|
| {py:class}`~cpm.applications.reinforcement_learning.RLRW` | {py:class}`~cpm.applications.reinforcement_learning.RLRWSession` |
| {py:class}`~cpm.applications.reinforcement_learning.HybridMBMF` | {py:class}`~cpm.applications.reinforcement_learning.HybridMBMFSession` |
| {py:class}`~cpm.applications.decision_making.PTSM` | {py:class}`~cpm.applications.decision_making.PTSMSession` |
| {py:class}`~cpm.applications.decision_making.PTSM1992` | {py:class}`~cpm.applications.decision_making.PTSM1992Session` |
| {py:class}`~cpm.applications.decision_making.PTSM2025` | {py:class}`~cpm.applications.decision_making.PTSM2025Session` |

A session version takes the same arguments, has the same parameters and priors, gives the same results and the same `export()`, and works with the optimisers, {py:class}`~cpm.generators.Simulator` and {py:mod}`cpm.hierarchical` in the same way. Swap the class and nothing else changes:

```python
from cpm.applications.reinforcement_learning import RLRWSession
from cpm.datasets import load_bandit_data
from cpm.optimisation import FminBound, minimise

data = load_bandit_data()
data["observed"] = data["response"]

model = RLRWSession(data=data[data.ppt == 1], dimensions=4)  # was: RLRW(...)
fit = FminBound(
    model=model,
    data=data.groupby("ppt"),
    minimisation=minimise.LogLikelihood.bernoulli,
    prior=True,
    number_of_starts=5,
    ppt_identifier="ppt",
    approx_grad=True,
)
fit.optimise()
```

## Install numba

The session versions run without numba, as plain Python, which is already 8 to 15 times faster than the per-trial applications. Compiled with numba, they are 3 to 16 times faster again. numba is an optional dependency:

```bash
pip install cpm-toolbox[numba]
```

numba supports a new NumPy release some time after it comes out, and does not support PyPy. If numba cannot be imported, for either reason, cpm uses the plain-Python version without further notice.

Choose between the two with `backend`:

- `backend="auto"` (the default) uses numba if it is available and plain Python otherwise.
- `backend="numba"` requires numba and raises an error without it.
- `backend="python"` never compiles. Use it to debug a model, or to check a result against the compiled version.

Both backends give the same results, to within rounding error.

## What to expect

One evaluation of the objective function, with the parameters' priors, for one participant (from `python benchmarks/run.py` in the cpm repository):

| Application | Trials | Per-trial | Session, Python | Session, numba |
|---|---:|---:|---:|---:|
| `RLRW` | 71 | 3.5 ms | 0.42 ms | 0.05 ms |
| `HybridMBMF` | 200 | 10 ms | 1.4 ms | 0.09 ms |
| `PTSM` | 40 | 2.0 ms | 0.22 ms | 0.06 ms |
| `PTSM1992` | 40 | 2.6 ms | 0.26 ms | 0.07 ms |
| `PTSM2025` | 40 | 2.2 ms | 0.15 ms | 0.06 ms |

With numba, the model itself takes 0.01 to 0.03 ms of this, and the loss function about 0.02 ms.

The first time a session model runs, numba compiles it, which takes 0.5 to 2.5 seconds. The machine code is cached on disk, next to cpm's own files, so later runs, new Python sessions and the worker processes of a parallel fit (see {doc}`parallelise-fitting`) load it in a fraction of a second. If cpm is installed in a directory you cannot write to, numba caches in your user directory instead.

## Simulations and random numbers

Where a model samples choices, as with `generate=True`, the session versions draw one uniform random number per trial from `numpy.random` before the loop and turn it into a choice exactly as `numpy.random.choice` does. A simulation after `numpy.random.seed(...)` therefore makes the same choices with the per-trial application, the session version and both backends.

## Where results can differ

The session versions compute softmax and logistic probabilities in a way that cannot overflow. Where the per-trial applications overflow (an inverse temperature times a value above about 709), they return NaN or replace it, and the two differ; everywhere else they agree to within rounding error.

Where a predicted probability comes within about $10^{-6}$ of 0 or 1, rounding errors of order $10^{-16}$ in the probability become relative errors of order $10^{-16}/(1 - p)$ in its log likelihood, in both versions. Their log likelihoods can then differ in the 8th or 10th significant digit instead of the 12th.

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

In a script or module, `@njit(cache=True)` keeps the compiled code on disk between Python sessions (not in an interactive session, where there is no file to cache it for).
{py:mod}`cpm.models.kernels` has function versions of the learning rules, decision rules, activation functions and attention mechanisms in {py:mod}`cpm.models`, written to be called from compiled loops like this one.

Inside a compiled function, use numbers, numpy arrays, and functions from `math`, `numpy` and {py:mod}`cpm.models.kernels`. pandas objects, dictionaries of mixed types, cpm's `Parameters` and `Value` objects, and most other Python objects are not available there, which is why the model function unpacks them first. See numba's [list of supported Python and NumPy features](https://numba.readthedocs.io/en/stable/reference/pysupported.html).

## Debug a compiled model

Set the environment variable `NUMBA_DISABLE_JIT=1` (or cpm's own `CPM_DISABLE_JIT=1`) before starting Python to run all compiled functions as plain Python, where you can use a debugger and `print` and get ordinary tracebacks. numba's error messages for code it cannot compile name the line and the types involved; the most common cause is passing a pandas object, a list or a `Value` into a compiled function.
