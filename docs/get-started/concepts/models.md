---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Models

A model in cpm is a Python function that describes **what happens on a single trial**.
cpm takes care of the rest: looping over trials, carrying the state of the model from one trial to the next, and collecting the output.

## The model function

The function takes two arguments:

- `parameters`: a {py:class}`~cpm.generators.Parameters` object with the parameters and the current state of the model (see {doc}`parameters`),
- `trial`: one row of the data, as a {py:class}`pandas.Series` (see {doc}`data-format`).

It returns a dictionary with everything you want to keep from the trial.
Here is a Rescorla–Wagner model with a Softmax choice rule, for the two-armed bandit data of the tutorials:

```{code-cell} ipython3
import numpy as np

from cpm.datasets import load_bandit_data
from cpm.generators import Parameters, Value
from cpm.models.decision import Softmax
from cpm.models.learning import SeparableRule

data = load_bandit_data()
data["observed"] = data["response"]

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

    return {
        "policy": choice_rule.policies,
        "values": values,
        "dependent": np.array([choice_rule.policies[1]]),
    }
```

Two kinds of keys in the output have a special meaning:

- **`dependent`** is the prediction of the model for the `observed` column of the data. The optimisers compare the two to fit the model.
- **keys with the same name as a parameter**, like `values` here, update that parameter, so the next trial starts from the updated state. This is how a model learns across trials.

All other keys are recorded.

## Running a model with a Wrapper

A {py:class}`~cpm.generators.Wrapper` runs the model function over all trials of one participant:

```{code-cell} ipython3
from cpm.generators import Wrapper

wrapper = Wrapper(model=model, parameters=parameters, data=data[data.ppt == 1])
wrapper.run()
wrapper.export().head()
```

{py:meth}`~cpm.generators.Wrapper.export` returns one row per trial, with one column per element of each output: `policy_0` and `policy_1`, `values_0` to `values_3`, and so on.
{py:meth}`~cpm.generators.Wrapper.reset` restores the initial state, optionally with new parameters or data.

## Running a model for many participants with a Simulator

A {py:class}`~cpm.generators.Simulator` runs a Wrapper for every participant.
It takes the data grouped by participant, and either one set of parameters for everyone, or one set per participant, for example the estimates of an optimiser:

```{code-cell} ipython3
import pandas as pd

from cpm.generators import Simulator

five = data[data.ppt <= 5]
per_participant = pd.DataFrame({"alpha": [0.1, 0.3, 0.5, 0.7, 0.9], "temperature": [5] * 5})

simulator = Simulator(wrapper=wrapper, parameters=per_participant, data=five.groupby("ppt"))
simulator.run()
simulator.export().groupby("ppt")[["values_0", "values_1", "values_2", "values_3"]].last()
```

## Building blocks

You rarely need to write the computations of a model from scratch.
{py:mod}`cpm.models` has the components of many common models, which you combine inside your model function:

| Module | Components |
| --- | --- |
| {py:mod}`cpm.models.learning` | learning rules: {py:class}`~cpm.models.learning.DeltaRule`, {py:class}`~cpm.models.learning.SeparableRule`, {py:class}`~cpm.models.learning.QLearningRule`, {py:class}`~cpm.models.learning.HumbleTeacher`, {py:class}`~cpm.models.learning.SARSATrace` |
| {py:mod}`cpm.models.decision` | choice rules: {py:class}`~cpm.models.decision.Softmax`, {py:class}`~cpm.models.decision.Sigmoid`, {py:class}`~cpm.models.decision.GreedyRule`, {py:class}`~cpm.models.decision.ChoiceKernel` |
| {py:mod}`cpm.models.activation` | activation functions: {py:class}`~cpm.models.activation.SigmoidActivation`, {py:class}`~cpm.models.activation.CompetitiveGating`, {py:class}`~cpm.models.activation.ProspectUtility`, {py:class}`~cpm.models.activation.Offset` |
| {py:mod}`cpm.models.attention` | attention: {py:class}`~cpm.models.attention.RapidAttentionShift` |
| {py:mod}`cpm.models.utils` | helpers: {py:func}`~cpm.models.utils.Nominal` |

For some common models, {py:mod}`cpm.applications` has ready-made Wrappers, such as {py:class}`~cpm.applications.reinforcement_learning.RLRW` and {py:class}`~cpm.applications.reinforcement_learning.HybridMBMF`.

Next: {doc}`fitting`.
