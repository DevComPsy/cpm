import itertools
import numpy as np
import pandas as pd

permute = np.array(list(itertools.permutations([1, 2, 3, 4], 2)))
repeated_permute = np.tile(permute, (5, 1))

rewards = np.zeros((4, 30))
rewards[0, 0:24] = 1
rewards[1, 0:6] = 1
rewards[2, 0:15] = 1
rewards[3, 27] = 1

indexing = repeated_permute.flatten()
actual_rewards = np.zeros(indexing.shape)

for i in np.unique(indexing):
    actual_rewards[indexing == i] = rewards[i - 1, np.where(repeated_permute == i)[1]]

actual_rewards = actual_rewards.reshape(repeated_permute.shape)
experiment = pd.concat(
    [pd.DataFrame(repeated_permute), pd.DataFrame(actual_rewards)], axis=1
)
experiment.columns = ["stimulus_left", "stimulus_right", "reward_left", "reward_right"]
experiment["ppt"] = 0

big = pd.DataFrame()
for i in range(100):
    current = experiment.sample(frac=1)
    current["ppt"] = i + 1
    big = pd.concat([big, current], axis=0)

big.reset_index(drop=True, inplace=True)
big.to_csv("bandit.csv", index=False)
