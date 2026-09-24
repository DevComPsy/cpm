---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Fitting models to data

The optimisers in {py:mod}`cpm.optimisation` estimate the free parameters of a model for every participant separately.
They take a {py:class}`~cpm.generators.Wrapper`, the data grouped by participant, and an objective function, and handle the rest: running the model, comparing its predictions with the data, and collecting the results.

```{code-cell} ipython3
:tags: [remove-cell]

# the model of the previous page, for 10 participants
import numpy as np

from cpm.datasets import load_bandit_data
from cpm.generators import Parameters, Value, Wrapper
from cpm.models.decision import Softmax
from cpm.models.learning import SeparableRule

np.random.seed(2026)
data = load_bandit_data()
data["observed"] = data["response"]
data = data[data.ppt <= 10]

parameters = Parameters(
    alpha=Value(value=0.5, lower=1e-10, upper=1, prior="truncated_normal", args={"mean": 0.5, "sd": 0.25}),
    temperature=Value(value=1, lower=0, upper=10, prior="truncated_normal", args={"mean": 5, "sd": 2.5}),
    values=np.array([0.25, 0.25, 0.25, 0.25]),
)


def model(parameters, trial):
    values = np.asarray(parameters.values).copy()
    stimuli = np.array([trial.arm_left, trial.arm_right]).astype(int)
    rewards = np.array([trial.reward_left, trial.reward_right])
    choice_rule = Softmax(activations=values[stimuli - 1], temperature=parameters.temperature)
    choice_rule.compute()
    choice = int(trial.response)
    chosen = np.zeros(4)
    chosen[stimuli[choice] - 1] = 1
    update = SeparableRule(weights=values, feedback=[rewards[choice]], input=chosen, alpha=parameters.alpha)
    update.compute()
    values += update.weights.flatten()
    return {"policy": choice_rule.policies, "values": values, "dependent": np.array([choice_rule.policies[1]])}


wrapper = Wrapper(model=model, parameters=parameters, data=data[data.ppt == 1])
```

We use the model and data of {doc}`models`, for 10 participants:

```{code-cell} ipython3
from cpm.optimisation import FminBound, minimise

fit = FminBound(
    model=wrapper,
    data=data.groupby("ppt"),
    minimisation=minimise.LogLikelihood.bernoulli,
    prior=True,
    number_of_starts=2,
    ppt_identifier="ppt",
    parallel=False,
    display=False,
    approx_grad=True,  # passed on to scipy.optimize.fmin_l_bfgs_b
)
fit.optimise()
```

The arguments that all optimisers share:

- `model`: the Wrapper. The **bounds** of the free parameters come from their {py:class}`~cpm.generators.Value` definitions, so you do not pass them again.
- `data`: the data, grouped by participant.
- `minimisation`: the objective function, see below.
- `prior`: with `True`, the optimiser minimises the negative log *posterior*, the negative log-likelihood plus the negative log prior, so the estimates are maximum *a posteriori*. With `False`, they are maximum-likelihood estimates.
- `number_of_starts`: how many random starting points to try per participant. The best fit is kept, which protects against local minima.
- `ppt_identifier`: the column that identifies participants, used to label the results.
- `parallel`: fit participants in parallel, see the how-to guide on {doc}`/how-to/parallelise-fitting`.
- `metrics`: extra measures of fit to compute at the estimates, such as {py:meth}`PenalisedLikelihoods.BIC <cpm.optimisation.compare.PenalisedLikelihoods.BIC>`.

All other keyword arguments are passed on to the underlying {py:mod}`scipy.optimize` function.

## The results

{py:meth}`~cpm.optimisation.FminBound.export` returns one row per participant:

```{code-cell} ipython3
fit.export().head()
```

- `x_0`, `x_1`, ... are the estimates of the free parameters, in the order of {py:meth}`~cpm.generators.Parameters.free`,
- `fun` is the value of the objective function at the estimates,
- `log_likelihood` and `log_prior` split it into its two parts when `prior=True`,
- the remaining columns come from the optimiser, such as the gradient and the Hessian.

`fit.parameters` holds the same estimates as a list with one dictionary per participant, which you can pass straight to a {py:class}`~cpm.generators.Simulator`:

```{code-cell} ipython3
fit.parameters[:3]
```

## Objective functions

{py:mod}`cpm.optimisation.minimise` has the objective functions.
They compare the `dependent` output of the model with the `observed` column of the data:

| Objective | Use it when the model predicts |
| --- | --- |
| {py:meth}`LogLikelihood.bernoulli <cpm.optimisation.minimise.LogLikelihood.bernoulli>` | the probability of a binary choice |
| {py:meth}`LogLikelihood.categorical <cpm.optimisation.minimise.LogLikelihood.categorical>` | the probabilities of more than two options |
| {py:meth}`LogLikelihood.continuous <cpm.optimisation.minimise.LogLikelihood.continuous>` | a continuous response |
| {py:class}`~cpm.optimisation.minimise.Distance` | a value to match, with squared-error measures |
| {py:class}`~cpm.optimisation.minimise.Discrete` | frequencies of discrete outcomes |

## Choosing an optimiser

| Optimiser | Method | Bounds | Hessian (for {py:mod}`cpm.hierarchical`) |
| --- | --- | --- | --- |
| {py:class}`~cpm.optimisation.FminBound` | L-BFGS-B, gradient-based | yes | yes |
| {py:class}`~cpm.optimisation.Fmin` | Nelder–Mead simplex | no | yes |
| {py:class}`~cpm.optimisation.Minimize` | any method of {py:func}`scipy.optimize.minimize` | passed on to the method | yes |
| {py:class}`~cpm.optimisation.DifferentialEvolution` | global, population-based | yes | no |
| {py:class}`~cpm.optimisation.Bads` | Bayesian adaptive direct search | yes | yes |

{py:class}`~cpm.optimisation.FminBound` is a good default: it is fast, respects the bounds, and works with the hierarchical methods.
If the likelihood surface has many local minima, try more starting points, or a global optimiser such as {py:class}`~cpm.optimisation.DifferentialEvolution`.
For noisy or expensive objective functions, {py:class}`~cpm.optimisation.Bads` is often more reliable.

## Next steps

Work through the {doc}`/tutorials/index`, starting with {doc}`/tutorials/first-model`.
