# Tutorials

The tutorials are a learning path through the core workflow of computational modelling with cpm.
Work through them in order: each one builds on the previous.
If you have never used cpm before, do the {doc}`/get-started/quickstart` first.

::::{grid} 1
:gutter: 3
:class-container: tutorial-path

:::{grid-item-card} 1. Build and fit your first model
:link: first-model
:link-type: doc

{bdg-success}`Beginner` {bdg-light}`30 min`
^^^
Build a reinforcement learning model of a two-armed bandit task from cpm components, fit it to data from many participants, and simulate new behaviour from the fitted parameters.
+++
{bdg-link-secondary}`Parameters <../api/generated/cpm.generators.Parameters.html>`
{bdg-link-secondary}`Wrapper <../api/generated/cpm.generators.Wrapper.html>`
{bdg-link-secondary}`FminBound <../api/generated/cpm.optimisation.FminBound.html>`
:::

:::{grid-item-card} 2. Parameter recovery
:link: parameter-recovery
:link-type: doc

{bdg-success}`Beginner` {bdg-light}`20 min`
^^^
Check whether the parameters of a model can be estimated reliably: simulate data from known parameters, fit the model, and compare.
+++
{bdg-link-secondary}`Simulator <../api/generated/cpm.generators.Simulator.html>`
{bdg-link-secondary}`LogLikelihood <../api/generated/cpm.optimisation.minimise.LogLikelihood.html>`
:::

:::{grid-item-card} 3. Model recovery and model comparison
:link: model-recovery
:link-type: doc

{bdg-warning}`Intermediate` {bdg-light}`30 min`
^^^
Check whether competing models can be told apart, compare them with penalised likelihoods, and map the landscape of their predictions.
+++
{bdg-link-secondary}`Simulator <../api/generated/cpm.generators.Simulator.html>`
{bdg-link-secondary}`PenalisedLikelihoods <../api/generated/cpm.optimisation.compare.PenalisedLikelihoods.html>`
:::

:::{grid-item-card} 4. Hierarchical estimation I: empirical Bayes
:link: hierarchical-empirical-bayes
:link-type: doc

{bdg-danger}`Advanced` {bdg-light}`40 min`
^^^
Estimate group-level priors from the data and use them to regularise the parameter estimates of each participant.
+++
{bdg-link-secondary}`EmpiricalBayes <../api/generated/cpm.hierarchical.EmpiricalBayes.html>`
:::

:::{grid-item-card} 5. Hierarchical estimation II: variational Bayes
:link: hierarchical-variational-bayes
:link-type: doc

{bdg-danger}`Advanced` {bdg-light}`40 min`
^^^
Estimate the group distribution of the parameters together with its uncertainty, and test group-level means.
+++
{bdg-link-secondary}`VariationalBayes <../api/generated/cpm.hierarchical.VariationalBayes.html>`
:::
::::

```{toctree}
:hidden:
:maxdepth: 1

first-model
parameter-recovery
model-recovery
hierarchical-empirical-bayes
hierarchical-variational-bayes
```
