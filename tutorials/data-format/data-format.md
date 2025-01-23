# Data formats

How should your data look like? `cpm` is optimised for specific data formats, which can be either a pandas.DataFrame or a list of dictionaries. 
For most models that use `cpm.generators.Wrapper`, it is a pandas.DataFrame, where each row should be a trial or state in the experiment or environment.
For `cpm.generators.Simulator` or methods in `cpm.optimisation`, the data should be a grouped pandas.DataFrame, where we fit the model to each groups.

If RAM is of concern, we recommend to use dictionaries. This is because dictionaries have smaller memory footprint than pandas.DataFrame.
Each dictionary represents a single participant. The key-value pairs in the dictionary are the names of the variables your model and the optimisation routines will need to simulate data and find the best-fitting parameters.

Imagine you have four stimuli in a given experiment, where on every trial they appear in a combination of two. Data then should be in the following format:

```python
participant = {
    "trials": np.array(
        [[2, 3],
         [1, 4],
         [3, 2],
         [4, 1],
         [2, 3],
         [2, 3],
         [1, 4],
         [3, 2],
         [4, 1],
         [2, 3]]
    ),
    "feedback": np.array([[1], [0], [1], [0], [1], [1], [0], [1], [0], [1]]),
    "observed": np.array([[1], [0], [1], [0], [1], [1], [0], [1], [0], [1]])
}
```

In this example, the `trials` key contains a list of all trials or states in the experiment or environment. Each trial is a list of two values, representing stimuli that appeared on that trial. The `feedback` key contains a list of feedback values, where `1` means the participant responded correctly and `0` means the participant responded incorrectly. The `observed` key contains a list of observed values, which are the same as the feedback values in this case. The only mandatory field that we need you to specify is the `observed` key, which must be your dependent variables.

**Note** that the first dimension of the keys in the dictionary indexes the trials! This is important because `cpm` will only loop through the first dimension!


You can also include additional variables in the data dictionary, such as participant id, reaction times. There is also no limits to what data you might include in the dictionary, as long as you have the `observed` key and everything else represents trial-by-trial information. What you include will largely depend on the model you are using.

Models in `cpm` will loop through each element in the keys and use the information to compute whatever is specified in the model. Note that, in `cpm`, we determine the number of trials/states by the length of the first dimension of the keys in the dictionary. `cpm` will take all the lengths of the keys in the dictionary and selects the maximum length as the number of trials/states.

## Convert your data from csv-type representation (pandas.DataFrame) to dictionaries

Most of us import data as a `pandas.DataFrame` from a csv. This usually looks something like this.

```python
      stimulus_left  stimulus_right  reward_left  reward_right  ppt
0                 4               3          0.0           1.0    1
1                 1               4          0.0           1.0    1
2                 4               2          0.0           0.0    1
3                 1               2          1.0           1.0    1
4                 4               3          0.0           0.0    1
...             ...             ...          ...           ...  ...
5995              1               2          1.0           0.0  100
5996              3               1          0.0           1.0  100
5997              4               2          1.0           0.0  100
5998              2               4          0.0           0.0  100
5999              1               3          1.0           1.0  100
```

`cpm` provides utilities to convert this type of representation to the right format.

```python
import pandas as pd
from cpm.utils import pandas_to_dict

dta = pd.read_csv("my_data.csv")
experiment = pandas_to_dict(dta, participant="ppt", stimuli='stim', feedback='reward', observed='response')
```

```python
[
    {
        "trials":
            array([[4, 3],
       [1, 4],
       [4, 2],
       [1, 2],
       [4, 3],
       [3, 4],
       [2, 3],
       [3, 4],
        ...
        ...
       [0., 1.],
       [0., 1.],
       [0., 1.],
       [1., 1.],
       [1., 0.],
       [1., 1.],
       [1., 1.],
       [0., 0.]]),
        "ppt"   : 1,
    },
]
```
