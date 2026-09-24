---
html_theme.sidebar_secondary.remove: true
---

# cpm

```{image} _static/logo/cpm-logo-light.png
:alt: cpm, the computational psychiatry modelling toolbox
:class: only-light hero-logo
:align: center
```

```{image} _static/logo/cpm-logo-dark.png
:alt: cpm, the computational psychiatry modelling toolbox
:class: only-dark hero-logo
:align: center
```

**cpm** is a Python toolbox for theory-driven computational modelling in psychiatry and psychology.
Build models from reusable components, fit them to data, check whether they can be trusted, and scale up to hundreds of participants.

```bash
pip install cpm-toolbox
```

::::{grid} 1 2 2 4
:gutter: 3
:class-container: landing-cards

:::{grid-item-card} {fas}`rocket` Get started
:link: get-started/index
:link-type: doc

Install cpm, fit your first model in five minutes, and learn the core concepts.
:::

:::{grid-item-card} {fas}`graduation-cap` Tutorials
:link: tutorials/index
:link-type: doc

A step-by-step path from fitting one model to hierarchical estimation.
:::

:::{grid-item-card} {fas}`images` Examples
:link: examples/index
:link-type: doc

Worked research examples: associative learning, two-step task, metacognition.
:::

:::{grid-item-card} {fas}`book` API reference
:link: api/index
:link-type: doc

Every class and function, with parameters, equations and references.
:::
::::

## Why cpm?

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item}
**Composable components.** Learning rules, decision rules and activation functions that you combine into your own models.
:::

:::{grid-item}
**One interface for every model.** {py:class}`~cpm.generators.Wrapper` and {py:class}`~cpm.generators.Simulator` run any model over trials and participants.
:::

:::{grid-item}
**Robust fitting.** Bounded and unbounded optimisers, differential evolution and BADS, with priors and multiple starts.
:::

:::{grid-item}
**Hierarchical estimation.** Empirical Bayes and variational Bayes for group-level priors.
:::

:::{grid-item}
**Ready-made models.** Rescorla–Wagner, hybrid model-based/model-free, prospect theory and meta-d′.
:::

:::{grid-item}
**Scales up.** Parallel fitting on your laptop, and job arrays on a computing cluster.
:::
::::

## Cite cpm

If you use cpm in your research, please cite:

> Dome, L., Hezemans, F. H., Kadri, K., Wagner, B. J., Webb, A., & Hauser, T. U. (2026).
> cpm: A Python library for theory-driven modelling in computational psychiatry.
> *PLOS Computational Biology*, 22(7), 1–31. <https://doi.org/10.1371/journal.pcbi.1014481>

See {doc}`about/citing` for the BibTeX entry.

```{toctree}
:hidden:

get-started/index
tutorials/index
examples/index
how-to/index
api/index
about/index
```
