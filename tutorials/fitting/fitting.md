# Fitting your models to data

Once you have specified your model and set up your parameters, you want to fit the model to your data.
`cpm` offers a set of optimisation techniques, a non-exhaustive list of some commonly used ones in Computational Psychiatry.
The main advantage of `cpm` here is that it takes the assumption that you want to fit the model to each subject's data separately, and that you want to fit the model to multiple subjects at once.
`cpm` covers most of the data cleaning and compilation steps for you, so it frees up time to do the more creative work.

## Optimisation

All optimisation routines already implemented will be found in the `cpm.optimisation` module, whereas `cpm.optimisation.minimise` handles all likelihood functions

```python
from cpm.optimisation import DifferentialEvolution, minimise

genetic = DifferentialEvolution(
    model=wrapped_model, # Wrapper class with the model we specified from before
    data=experiment, # the data as a list of dictionaries
    minimisation=minimise.LogLikelihood.bernoulli,
    parallel=True,
    ppt_identifier="ppt",
    display=False,
    maxiter=400, # additional arguments passed to scipy.optimise.differential_evolution
    tol=1e-10, # additional arguments passed to scipy.optimise.differential_evolution
)
```

There are a few things to note here.
First, you don't define bounds for the parameter space.
The reason for this is that you already defined bounds in the Parameter class for your model.
Bounds are part of the model specification and you will have to explicitly state them when you are building the model.
The second is that everything at the end that is not part of the named list of arguments for `cpm.optimisation.DifferentialEvolution` will be passed on the `scipy.optimize.differential_evolution` function.
`cpm` tries to avoid implementing methods from scratch, which is why we use `scipy` for the optimisation routines, and write most of the codebase with the intent on handling data cleaning and processing challenges.

## Parallel computing on Windows

Issue[#16](https://github.com/DevComPsy/modelling-toolbox/issues/16) is a known issue with Windows and parallelisation.
The way to circumvent this is to wrap your code in into `if __name__ == '__main__':` and load the parallelised optimisation modules under that code. Here is an example:

```python
import pandas as pd
import numpy as np
import time
import os
import cpm

from cpm.generators import Parameters, Wrapper
from cpm.models import learning, decision
from cpm.optimisation import minimise

if __name__ == '__main__:
    from cpm.optimisation import Fmin
    # the rest of the code goes here...

```