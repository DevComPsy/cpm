# How-to guides

Short, task-focused recipes for problems you will meet once you have learned the basics.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {fas}`bolt` Speed up fitting with session models
:link: fast-session-models
:link-type: doc

Fit the built-in models 50 to 200 times faster, and compile your own models with numba.
:::

:::{grid-item-card} {fas}`gauge-high` Parallelise model fitting
:link: parallelise-fitting
:link-type: doc

Fit many participants at once on a single machine, including on Windows and in Jupyter.
:::

:::{grid-item-card} {fas}`server` Run fits on a computing cluster
:link: run-on-hpc
:link-type: doc

Split participants across the nodes of a SLURM cluster with a job array, then combine the results.
:::
::::

```{toctree}
:hidden:

fast-session-models
parallelise-fitting
run-on-hpc
```
