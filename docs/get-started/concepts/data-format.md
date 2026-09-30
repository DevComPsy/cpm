---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Data format

cpm works with [pandas](https://pandas.pydata.org/) data frames in which **each row is one trial** of one participant.
The columns hold everything the model needs to know about a trial: which stimuli were shown, what the rewards were, what the participant chose.

```{code-cell} ipython3
import numpy as np
import pandas as pd

from cpm.datasets import load_bandit_data

data = load_bandit_data()
data.head()
```

There are no required column names, apart from one: the column **`observed`** holds the behaviour that the model should predict, and that the optimisers compare the predictions of the model with.
Your model function reads the other columns by name, so you choose them.

```{code-cell} ipython3
data["observed"] = data["response"]
```

## One participant or many

A {py:class}`~cpm.generators.Wrapper` runs a model over the trials of **one** participant, so it takes the rows of a single participant:

```{code-cell} ipython3
one_participant = data[data.ppt == 1]
one_participant.shape
```

A {py:class}`~cpm.generators.Simulator` and the optimisers in {py:mod}`cpm.optimisation` work on **many** participants.
They take the data grouped by participant, with {py:meth}`pandas.DataFrame.groupby`, and run the model on each group:

```{code-cell} ipython3
grouped = data.groupby("ppt")
print(f"{grouped.ngroups} participants")
```

The optimisers also need to know which column identifies participants, so that they can label their results: pass its name as `ppt_identifier`.

## Values that change within a trial

Each cell of a data frame holds one value.
If a trial has several values of the same kind, such as the stimuli on screen, give each its own column (`arm_left`, `arm_right`), and combine them in the model function:

```{code-cell} ipython3
trial = one_participant.iloc[0]
stimuli = np.array([trial.arm_left, trial.arm_right])
stimuli
```

## Data for the ready-made models

The models in {py:mod}`cpm.applications` expect specific column names, which are listed on their API pages.
{py:mod}`cpm.utils.data` has converters that rename and reshape your columns for them, such as {py:func}`~cpm.utils.data.convert_to_RLRW` for {py:class}`~cpm.applications.reinforcement_learning.RLRW`:

```{code-cell} ipython3
from cpm.utils.data import convert_to_RLRW

my_data = pd.DataFrame(
    {
        "subject": [1, 1, 2],
        "stim_left": [1, 2, 1],
        "stim_right": [2, 1, 2],
        "choice": [1, 0, 1],
        "reward_left": [1, 0, 1],
        "reward_right": [0, 1, 0],
    }
)
convert_to_RLRW(
    data=my_data,
    human_response="choice",
    reward=["reward_left", "reward_right"],
    stimulus=["stim_left", "stim_right"],
    participant="subject",
)
```

## Example datasets

{py:mod}`cpm.datasets` ships with the datasets used in the tutorials and examples:

| Loader | Task |
| --- | --- |
| {py:func}`~cpm.datasets.load_bandit_data` | two-armed bandit with four stimuli |
| {py:func}`~cpm.datasets.load_two_step_data` | deterministic two-step task (Smid et al., 2022) |
| {py:func}`~cpm.datasets.load_risky_choices` | risky decision making with gains, losses and ambiguity |
| {py:func}`~cpm.datasets.load_metacognition_data` | perceptual decisions with confidence ratings |

Next: {doc}`parameters`.
