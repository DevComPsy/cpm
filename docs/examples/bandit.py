import itertools
import numpy as np
import pandas as pd

permute = np.array(list(itertools.permutations([1, 2, 3, 4], 2)))
repeated_permute = np.tile(permute, (5, 1))

rewards = np.zeros((4, 30))
rewards[0, 0:24] = 1
rewards[1, 0:6] = 1
rewards[2, 0:15] = 1
rewards[3, 0:3] = 1

probabilities = np.array([24 / 30, 6 / 30, 15 / 30, 3 / 30])

indexing = repeated_permute.flatten()
actual_rewards = np.zeros(indexing.shape)

for i in np.unique(indexing):
    np.random.shuffle(rewards[i - 1])
    actual_rewards[indexing == i] = rewards[i - 1]

actual_rewards = actual_rewards.reshape(repeated_permute.shape)
experiment = pd.concat(
    [pd.DataFrame(repeated_permute), pd.DataFrame(actual_rewards)], axis=1
)
experiment.columns = ["stimulus_left", "stimulus_right", "reward_left", "reward_right"]
experiment["ppt"] = 0

experiment = pd.concat([experiment, experiment]).reset_index(drop=True)

big = pd.DataFrame()
for i in range(100):
    current = experiment.sample(frac=1)
    current["ppt"] = i + 1
    choices = np.zeros(current.shape[0])
    for x in range(current.shape[0]):
        left = current.stimulus_left[x]
        right = current.stimulus_right[x]
        stimuli = np.array([left, right])
        p_normal = probabilities[stimuli - 1] / np.sum(probabilities[stimuli - 1])
        choices[x] = np.argmax(p_normal)
    accuracy = 1
    while accuracy >= 1:
        accuracy = np.abs(np.random.normal(0.25, 0.25))
    print(accuracy)
    error_rates = np.random.binomial(1, p=accuracy, size=choices.shape[0]).astype(bool)
    choices[error_rates] = 1 - choices[error_rates]
    current["trial"] = np.arange(choices.shape[0]) + 1
    current["responses"] = choices.astype(int)
    big = pd.concat([big, current], axis=0)

big.groupby(["stimulus_left", "stimulus_right"]).mean()

big.reset_index(drop=True, inplace=True)
big.to_csv("bandit.csv", index=False)
