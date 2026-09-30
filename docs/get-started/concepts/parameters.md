---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Parameters and priors

The parameters of a model are described with {py:class}`~cpm.generators.Parameters`.
It holds two kinds of things:

- **free parameters**, which are estimated from the data. Each is a {py:class}`~cpm.generators.Value` with a starting value, lower and upper bounds, and a prior distribution.
- **everything else the model needs to start**, such as the initial values of the stimuli. These are plain numbers or arrays, without a prior.

```{code-cell} ipython3
import numpy as np

from cpm.generators import Parameters, Value

parameters = Parameters(
    alpha=Value(
        value=0.5,
        lower=1e-10,
        upper=1,
        prior="truncated_normal",
        args={"mean": 0.5, "sd": 0.25},
    ),
    temperature=Value(
        value=1,
        lower=0,
        upper=10,
        prior="truncated_normal",
        args={"mean": 5, "sd": 2.5},
    ),
    values=np.array([0.25, 0.25, 0.25, 0.25]),
)
```

**A parameter is free if, and only if, it has a prior.**
{py:meth}`~cpm.generators.Parameters.free` lists the free parameters, and {py:meth}`~cpm.generators.Parameters.bounds` their bounds, which the optimisers use:

```{code-cell} ipython3
parameters.free()
```

```{code-cell} ipython3
parameters.bounds()
```

Inside a model function, parameters behave like numbers:

```{code-cell} ipython3
parameters.alpha * 2
```

## Priors

`prior` is the name of a distribution, and `args` its arguments.
The distributions are those of {py:mod}`scipy.stats`:

| `prior` | `args` |
| --- | --- |
| `"uniform"` | none: uniform between `lower` and `upper` |
| `"truncated_normal"` | `mean`, `sd` |
| `"norm"` | `mean`, `sd` |
| `"beta"` | `a`, `b`, and `mean` and `sd` as its location and scale |
| `"gamma"` | `a`, and `mean` and `sd` as its location and scale |
| `"truncated_exponential"` | `mean` and `sd` as its location and scale, truncated at `upper` |

You can also pass a {py:mod}`scipy.stats` distribution itself, such as `scipy.stats.multivariate_normal`: cpm calls it with `args` as keyword arguments.

The priors are used in three places:

- when fitting with `prior=True`, the optimisers maximise the posterior instead of the likelihood,
- {py:meth}`~cpm.generators.Parameters.sample` draws parameters from their priors, for example to simulate data,
- hierarchical methods ({py:mod}`cpm.hierarchical`) update the priors to the estimated group distribution.

```{code-cell} ipython3
parameters.sample(size=3)
```

{py:meth}`~cpm.generators.Parameters.PDF` gives the joint prior density of the current values, or its logarithm with `log=True`:

```{code-cell} ipython3
parameters.PDF(log=True)
```

A density can be larger than 1: unlike a probability, it is only bounded by the requirement that it integrates to 1 over the whole range of the parameter.

## Updating priors

{py:meth}`~cpm.generators.Parameters.update_prior` changes the arguments of the priors, one keyword argument per parameter:

```{code-cell} ipython3
parameters.update_prior(alpha={"mean": 0.3, "sd": 0.1})
parameters.alpha.prior.mean()
```

## Parameters on a transformed scale

Some optimisers work better on an unbounded scale.
{py:class}`~cpm.generators.LogParameters` stores the parameters on a logit scale within their bounds, and transforms them back for the model.

Next: {doc}`models`.
