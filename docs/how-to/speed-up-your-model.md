# Speed up your own model

The built-in models in {py:mod}`cpm.applications` are already fast.
If fitting your own model is too slow, usually because you fit it many times (in hierarchical estimation, parameter recovery or model comparison), write its model function for all trials at once, and wrap it in a {py:class}`~cpm.generators.SessionWrapper` instead of a {py:class}`~cpm.generators.Wrapper`.
Everything else stays the same: the optimisers, {py:class}`~cpm.generators.Simulator`, {py:mod}`cpm.hierarchical` and `export()` work with either.

For the model on this page, fitted to one participant of the bandit data (71 trials), one evaluation of the objective function takes:

| Model function | Time per evaluation |
|---|---:|
| Trial by trial, in a `Wrapper` | 4.2 ms |
| All trials at once, in a `SessionWrapper` | 0.44 ms |
| All trials at once, compiled with numba | 0.06 ms |

A fit evaluates the objective function thousands of times per participant, and hierarchical estimation refits every participant on each iteration.

## The same model, two ways

This is the model of {doc}`/tutorials/first-model`, trial by trial:

```python
import numpy as np

from cpm.datasets import load_bandit_data
from cpm.generators import Parameters, Value
from cpm.models.decision import Softmax
from cpm.models.learning import SeparableRule

experiment = load_bandit_data()
experiment["observed"] = experiment["response"]

parameters = Parameters(
    alpha=Value(value=0.5, lower=1e-10, upper=1, prior="truncated_normal", args={"mean": 0.5, "sd": 0.25}),
    temperature=Value(value=1, lower=0, upper=10, prior="truncated_normal", args={"mean": 5, "sd": 2.5}),
    values=np.array([0.25, 0.25, 0.25, 0.25]),
)


def model(parameters, trial, generate=False):
    alpha = parameters.alpha
    temperature = parameters.temperature
    values = np.asarray(parameters.values).copy()

    # the stimuli on screen, and the reward each of them would give
    stimuli = np.array([trial.arm_left, trial.arm_right]).astype(int)
    rewards = np.array([trial.reward_left, trial.reward_right])

    # Equation 1: choice probabilities from the values of the two stimuli
    # (stimuli are numbered from 1, Python counts from 0)
    choice_rule = Softmax(activations=values[stimuli - 1], temperature=temperature)
    choice_rule.compute()

    # learn from the choice of the participant, or from the model's own choice
    choice = choice_rule.choice() if generate else int(trial.response)
    reward = rewards[choice]

    # Equation 2: update the value of the chosen stimulus
    chosen = np.zeros(4)
    chosen[stimuli[choice] - 1] = 1
    update = SeparableRule(weights=values, feedback=[reward], input=chosen, alpha=alpha)
    update.compute()
    values += update.weights.flatten()

    return {
        "policy": choice_rule.policies,  # probability of choosing left and right
        "response": choice,  # the choice the model learned from
        "reward": reward,
        "error": update.error[0, stimuli[choice] - 1],  # the prediction error
        "values": values,  # the updated values, used on the next trial
        "dependent": np.array([choice_rule.policies[1]]),  # what the model is fitted to
    }
```

And the same model for all trials at once:

```python
def session_model(parameters, data, generate=False):
    alpha = parameters.alpha.value
    temperature = parameters.temperature.value
    values = np.asarray(parameters.values, dtype=float).copy()

    # one row per trial for every output
    n = len(data["response"])
    policy = np.empty((n, 2))
    response = np.empty(n, dtype=int)
    reward = np.empty(n)
    error = np.empty(n)
    history = np.empty((n, 4))

    for t in range(n):
        stimuli = np.array([data["arm_left"][t], data["arm_right"][t]]).astype(int) - 1
        rewards = np.array([data["reward_left"][t], data["reward_right"][t]])

        # Equation 1
        exponentials = np.exp(values[stimuli] * temperature)
        policy[t] = exponentials / exponentials.sum()

        choice = np.random.choice(2, p=policy[t]) if generate else int(data["response"][t])
        response[t] = choice
        reward[t] = rewards[choice]

        # Equation 2
        error[t] = reward[t] - values[stimuli[choice]]
        values[stimuli[choice]] += alpha * error[t]
        history[t] = values

    return {
        "policy": policy,
        "response": response,
        "reward": reward,
        "error": error,
        "values": history,
        "dependent": policy[:, 1],
    }
```

The differences:

- **The loop over trials is inside the model function.** A `Wrapper` calls `model(parameters, trial)` once per trial; a `SessionWrapper` calls `model(parameters, data)` once per run.
- **`data` replaces `trial`.** It is a dictionary of numpy arrays, one per column of the data, so `trial.arm_left` becomes `data["arm_left"][t]`.
- **The formulas are written out.** Equations 1 and 2 are computed with numpy, instead of creating a `Softmax` and a `SeparableRule` on every trial. `np.random.choice(2, p=...)` is how `Softmax.choice()` samples, so simulations make the same choices.
- **States are local variables.** `values` is updated in the loop and recorded on every trial. After a run, the `SessionWrapper` sets the `values` parameter to its last row, the state a `Wrapper` ends a run with.
- **Outputs are arrays**, one row per trial. `dependent` is what the model is fitted to, as before, and `export()` gives the same table as with a `Wrapper`.

Wrap the new model function in a `SessionWrapper`, and use it wherever you used the `Wrapper`:

```python
from cpm.generators import SessionWrapper, Wrapper

one = experiment[experiment.ppt == 1]
per_trial = Wrapper(model=model, parameters=parameters, data=one)
all_trials = SessionWrapper(model=session_model, parameters=parameters, data=one)
```

## Check that both give the same results

A mistake in the loop changes the results without raising an error, so compare the new model function with the per-trial one before you use it, at a few parameter values:

```python
for alpha, temperature in [(0.1, 1.0), (0.5, 5.0), (0.9, 9.0)]:
    for wrapper in (per_trial, all_trials):
        wrapper.reset(parameters={"alpha": alpha, "temperature": temperature})
        wrapper.run()
    print(np.allclose(per_trial.dependent, all_trials.dependent))
```

All three lines should print `True`.
If your model simulates choices, also compare the exports of a simulation with each version, each run after the same `np.random.seed(...)`.

## Optional: compile it with numba

With [numba](https://numba.pydata.org/) installed (`pip install "cpm-toolbox[numba]"`, see {doc}`fast-session-models`), the loop can be compiled to machine code.
Move the loop into a function of numbers and arrays only, decorate it with `njit`, and call it from the model function:

```python
from numba import njit


@njit(cache=True)
def learn(alpha, temperature, values, arm_left, arm_right, reward_left, reward_right, response):
    values = values.copy()
    n = response.shape[0]
    p_right = np.empty(n)
    for t in range(n):
        left, right = arm_left[t] - 1, arm_right[t] - 1

        # Equation 1
        exponentials = np.exp(np.array([values[left], values[right]]) * temperature)
        p_right[t] = exponentials[1] / exponentials.sum()

        # Equation 2
        chosen = right if response[t] == 1 else left
        reward = reward_right[t] if response[t] == 1 else reward_left[t]
        values[chosen] += alpha * (reward - values[chosen])
    return p_right


def compiled_model(parameters, data):
    p_right = learn(
        float(parameters.alpha), float(parameters.temperature), np.asarray(parameters.values, dtype=float),
        data["arm_left"], data["arm_right"], data["reward_left"], data["reward_right"], data["response"],
    )
    return {"p_right": p_right, "dependent": p_right}


compiled = SessionWrapper(model=compiled_model, parameters=parameters, data=one)
```

It computes what the loop above computes, with numbers and arrays only, and without the outputs this version does not need.
Inside a compiled function, use numbers, numpy arrays, and functions from `math` and `numpy`; pandas objects, dictionaries and cpm's `Parameters` are not available there, which is why the model function unpacks them first.
This version only fits the model; for simulations, use the one above.

The first call compiles the function, which takes a second or two.
With `cache=True`, numba keeps the compiled code on disk, and later Python sessions load it in a fraction of a second.
This works in scripts and in Jupyter notebooks; at the plain Python prompt, numba cannot cache and raises an error, so leave out `cache=True` there.
To find errors in a compiled function, see [Debug a compiled model](fast-session-models.md#debug-a-compiled-model).
