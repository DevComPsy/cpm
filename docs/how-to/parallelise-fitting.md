# Parallelise model fitting

The optimisers in {py:mod}`cpm.optimisation` fit each participant independently, so they can fit several participants at the same time, one per core.

## Turning it on

Pass `parallel=True`, and the number of cores to use as `cl`:

```python
from cpm.optimisation import FminBound, minimise

fit = FminBound(
    model=wrapper,
    data=data.groupby("ppt"),
    minimisation=minimise.LogLikelihood.bernoulli,
    prior=True,
    number_of_starts=5,
    ppt_identifier="ppt",
    parallel=True,
    cl=8,  # the number of cores; all cores if None
    approx_grad=True,
)
fit.optimise()
```

The results have the same format, with the participants in the same order; with random starting points, the estimates can differ slightly between runs, as they do without `parallel=True`.
{py:class}`~cpm.applications.signal_detection.EstimatorMetaD` takes the same `parallel` and `cl` arguments.

## In a script or in a notebook

cpm picks one of two backends, depending on where your code runs:

- In a Python **script**, it uses [multiprocess](https://github.com/uqfoundation/multiprocess), which starts one worker process per core.
- In a **Jupyter notebook**, it uses [ipyparallel](https://ipyparallel.readthedocs.io/), which starts a small cluster of engines for each fit. The engines need to import the libraries that your model uses; list them in `libraries`, for example `libraries=["numpy", "pandas", "cpm"]`. Starting the engines takes a few seconds, so parallel fitting only pays off in a notebook when fitting takes much longer than that.

## On Windows and macOS: use a main guard

On Windows and macOS, new processes start a fresh Python interpreter that imports your script again.
Without a guard, every worker would start fitting again itself.
Put the code that runs the fit under `if __name__ == "__main__":`

```python
from cpm.optimisation import FminBound, minimise


def main():
    data = ...
    wrapper = ...
    fit = FminBound(model=wrapper, data=data.groupby("ppt"), parallel=True)  # and your other arguments
    fit.optimise()
    fit.export().to_csv("fits.csv", index=False)


if __name__ == "__main__":
    main()
```

Define your model function at the top level of a module, not inside another function, so that the workers can find it.

## How many cores?

- More cores than participants does not help: each participant is fitted on one core.
- Each worker holds a copy of the data of its participant and of the model, so on a machine with little memory, use fewer cores.
- If Windows reports *"The paging file is too small for this operation to complete"*, it has run out of virtual memory for the new processes: use fewer cores, or increase the size of the paging file.

## Across the nodes of a cluster

To spread fits over several machines, see {doc}`run-on-hpc`.
