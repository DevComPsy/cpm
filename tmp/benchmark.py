import argparse
import timeit

PARALELL_TEMPLATE = """
from cpm.utils import pandas_to_dict
import numpy as np
import pandas as pd

from src.fitting import fitting
from src.model import model_setup

data = pd.read_csv("simulated_data.csv")

experiment = pandas_to_dict(
    data,
    participant="ppt",
    stimuli="stimulus",
    feedback="reward",
    observed="responses",
    trial_number="trial",
)
sd_start = np.random.uniform(0.1, 1, 2) * np.array([1, 10])
mean_start = np.random.uniform(0, 1, 2) * np.array([1, 10])
parameters, learning_model = model_setup(
    experiment[0], mean=mean_start, sd=sd_start, generate=False
)
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-rs", "--repeats", type=int, default=5)

    args = parser.parse_args()

    cmd = "fitting(wrapper=learning_model, data=experiment, parallel=True)"

    timeit.timeit(
        cmd,
        setup=PARALELL_TEMPLATE,
        number=args.repeats,
    )
