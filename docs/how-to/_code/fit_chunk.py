"""Fit the participants of one chunk of the data, as one task of a SLURM job array."""

import os
from pathlib import Path

import numpy as np

from cpm.applications.reinforcement_learning import RLRW
from cpm.datasets import load_bandit_data
from cpm.optimisation import FminBound, minimise


def main():
    # which task of the job array this is, and how many tasks there are
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    data = load_bandit_data()
    data["observed"] = data["response"]

    # split the participants into one chunk per task, and keep this task's chunk
    chunks = np.array_split(np.sort(data.ppt.unique()), tasks)
    chunk = data[data.ppt.isin(chunks[task])]
    print(f"task {task + 1} of {tasks}: {chunk.ppt.nunique()} participants on {cores} cores")

    model = RLRW(data=chunk[chunk.ppt == chunk.ppt.iloc[0]], dimensions=4, parameters_settings=[[0.5, 0, 1], [5, 0, 10]])
    fit = FminBound(
        model=model,
        data=chunk.groupby("ppt"),
        minimisation=minimise.LogLikelihood.bernoulli,
        prior=True,
        number_of_starts=5,
        ppt_identifier="ppt",
        parallel=cores > 1,
        cl=cores,
        display=False,
        approx_grad=True,
    )
    fit.optimise()

    results = Path("results")
    results.mkdir(exist_ok=True)
    fit.export().to_csv(results / f"fit_{task:03d}.csv", index=False)


# the guard is required for parallel fitting, see the how-to guide on parallel fitting
if __name__ == "__main__":
    main()
