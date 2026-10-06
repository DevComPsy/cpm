"""
Where does the time of a hierarchical fit go?

    python benchmarks/hierarchical_profile.py [--data DATASET.csv] [--iterations 5] [--label NAME]

Fits one dataset of the four-armed bandit (100 participants, 120 trials) with
`RLRW` and `VariationalBayes` for a fixed number of iterations, one chain and
two optimiser starts per participant, under `cProfile`. Prints the share of
time in the optimiser, the Hessian, `evaluate_fit`, the population update and
the parts of `objective`, and the number of objective evaluations per
participant and iteration, separately for the optimiser, the Hessian and
`evaluate_fit`. Writes the table to benchmarks/results/<label>.csv.

`--data` is a dataset in the format `RLRW` reads (columns ppt, trial, arm_left,
arm_right, reward_left, reward_right, response). Without it, the task and the
responses are simulated from a known population. The settings match the speed
comparison with cbm: bounds alpha [0.01, 1] and temperature [0, 10], a starting
prior of mean 0.5 and sd 0.5 for alpha and mean 5 and sd 5 for temperature.

The profiler adds overhead to Python-heavy code, so the shares are indicative;
the fit is also timed once without it.
"""

import argparse
import cProfile
import pathlib
import pstats
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import cpm  # noqa: E402
import cpm.optimisation.fmin as fmin_module  # noqa: E402
from cpm.applications.reinforcement_learning import RLRW  # noqa: E402
from cpm.hierarchical import VariationalBayes  # noqa: E402
from cpm.optimisation import FminBound, minimise  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
PARTICIPANTS, TRIALS, STIMULI = 100, 120, 4
BOUNDS = {"alpha": (1e-2, 1.0), "temperature": (0.0, 10.0)}
STARTING_PRIOR = {"alpha": {"mean": 0.5, "sd": 0.5}, "temperature": {"mean": 5.0, "sd": 5.0}}
POPULATION = {"alpha": {"mean": 0.5, "sd": 0.15}, "temperature": {"mean": 4.5, "sd": 1.5}}


def simulate(seed=2026):
    """A random four-armed bandit task with responses of `RLRW` from a known population."""
    rng = np.random.default_rng(seed)
    n, trials = PARTICIPANTS, TRIALS
    arms = np.array([rng.choice(STIMULI, size=2, replace=False) for _ in range(n * trials)]).reshape(n, trials, 2)
    reward_probability = np.array([0.2, 0.4, 0.6, 0.8])
    rewards = (rng.random((n, trials, 2)) < reward_probability[arms]).astype(float)
    parameters = {
        name: truncnorm.rvs(
            (BOUNDS[name][0] - p["mean"]) / p["sd"], (BOUNDS[name][1] - p["mean"]) / p["sd"],
            loc=p["mean"], scale=p["sd"], size=n, random_state=rng,
        )
        for name, p in POPULATION.items()
    }
    values = np.full((n, STIMULI), 1 / STIMULI)
    rows = np.arange(n)
    response = np.empty((n, trials), dtype=int)
    for t in range(trials):
        scaled = values[rows[:, None], arms[:, t]] * parameters["temperature"][:, None]
        scaled -= scaled.max(axis=1, keepdims=True)
        policy = np.exp(scaled)
        policy /= policy.sum(axis=1, keepdims=True)
        choice = (rng.random(n) >= policy[:, 0]).astype(int)
        stimulus = arms[rows, t, choice]
        values[rows, stimulus] += parameters["alpha"] * (rewards[rows, t, choice] - values[rows, stimulus])
        response[:, t] = choice
    return pd.DataFrame(
        {
            "ppt": np.repeat(np.arange(1, n + 1), trials),
            "trial": np.tile(np.arange(1, trials + 1), n),
            "arm_left": arms[:, :, 0].ravel() + 1,
            "arm_right": arms[:, :, 1].ravel() + 1,
            "reward_left": rewards[:, :, 0].ravel(),
            "reward_right": rewards[:, :, 1].ravel(),
            "response": response.ravel(),
            "observed": response.ravel(),
        }
    )


def load(path):
    data = pd.read_csv(path)
    ppts = np.sort(data.ppt.unique())[:PARTICIPANTS]
    data = data[data.ppt.isin(ppts) & (data.trial <= TRIALS)].reset_index(drop=True)
    data["observed"] = data["response"]
    return data


def setup(data, iterations, **options):
    """`VariationalBayes` with the settings of the speed comparison, forced to run `iterations` iterations; `options` go to `FminBound`."""
    first = data[data.ppt == data.ppt.iloc[0]]
    model = RLRW(data=first, dimensions=STIMULI, parameters_settings=[[0.5, *BOUNDS["alpha"]], [2.0, *BOUNDS["temperature"]]])
    model.parameters.update_prior(**STARTING_PRIOR)
    optimiser = FminBound(
        model=model,
        data=data.groupby("ppt"),
        minimisation=minimise.LogLikelihood.bernoulli,
        prior=True,
        number_of_starts=2,
        ppt_identifier="ppt",
        parallel=False,
        display=False,
        approx_grad=True,
        **options,
    )
    return VariationalBayes(
        optimiser=optimiser,
        iteration=iterations,
        chain=1,
        tolerance_lme=0.0,
        tolerance_param=0.0,
        hyperpriors={"a0": np.array([p["mean"] for p in STARTING_PRIOR.values()]), "b": 1, "v": 0.5, "s": np.repeat(0.01, 2)},
        quiet=True,
    )


class Counter:
    """Counts the objective evaluations of `FminBound` by what asked for them."""

    def __init__(self):
        self.phase = "optimiser"
        self.counts = {"optimiser": 0, "hessian": 0, "evaluate_fit": 0}
        self.originals = (fmin_module.objective, fmin_module.numerical_hessian, fmin_module.finite_difference_hessian,
                          fmin_module.evaluate_fit)

    def __enter__(self):
        objective, numerical_hessian, finite_difference_hessian, evaluate_fit = self.originals

        def counted_objective(*args, **kwargs):
            self.counts[self.phase] += 1
            return objective(*args, **kwargs)

        def counted(hessian):
            def counted_hessian(*args, **kwargs):
                self.phase = "hessian"
                try:
                    return hessian(*args, **kwargs)
                finally:
                    self.phase = "optimiser"

            return counted_hessian

        def counted_evaluate_fit(*args, **kwargs):
            self.counts["evaluate_fit"] += 1  # one model run per call
            return evaluate_fit(*args, **kwargs)

        fmin_module.objective = counted_objective
        fmin_module.numerical_hessian = counted(numerical_hessian)
        fmin_module.finite_difference_hessian = counted(finite_difference_hessian)
        fmin_module.evaluate_fit = counted_evaluate_fit
        return self

    def __exit__(self, *exc):
        (fmin_module.objective, fmin_module.numerical_hessian, fmin_module.finite_difference_hessian,
         fmin_module.evaluate_fit) = self.originals


def profile(vb):
    profiler = cProfile.Profile()
    profiler.enable()
    vb.optimise()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.calc_callees()
    return stats


def find(stats, filename, function):
    """The (file, line, name) key of a profiled function, by the basename of its file."""
    for key in stats.stats:
        if key[2] == function and pathlib.Path(key[0]).name == filename:
            return key
    return None


def inclusive(stats, filename, function):
    key = find(stats, filename, function)
    return stats.stats[key][3] if key else 0.0


def inclusive_from(stats, caller, filename, function):
    """Seconds spent in `function` when called from `caller`."""
    callees = stats.all_callees.get(caller, {})
    return sum(value[3] for key, value in callees.items() if key[2] == function and pathlib.Path(key[0]).name == filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="a dataset CSV; simulated if not given")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--label", default="hierarchical-profile")
    parser.add_argument("--hessian", default="numdifftools", help="the `hessian` option of FminBound")
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    data = load(args.data) if args.data else simulate()
    participants, iterations = data.ppt.nunique(), args.iterations
    options = {"hessian": args.hessian}
    print(f"cpm {cpm.__version__}: {participants} participants, {data.trial.max()} trials, {iterations} iterations, 2 starts, {options}")

    # the compiled model and everything else that loads on the first call
    setup(data[data.ppt.isin(data.ppt.unique()[:5])], 1, **options).optimise()

    # once without the profiler, for the time
    np.random.seed(2026)
    start = time.perf_counter()
    with Counter() as counter:
        setup(data, iterations, **options).optimise()
    seconds = time.perf_counter() - start
    per = participants * iterations
    counts = {k: v / per for k, v in counter.counts.items()}

    np.random.seed(2026)
    stats = profile(setup(data, iterations, **options))
    total = inclusive(stats, "variational.py", "optimise")
    objective_key = find(stats, "optimisers.py", "objective")
    rows = [
        ("fit", "seconds without profiler", seconds),
        ("fit", "ms per participant and iteration", seconds * 1e3 / per),
        ("evaluations per participant and iteration", "optimiser", counts["optimiser"]),
        ("evaluations per participant and iteration", "hessian", counts["hessian"]),
        ("evaluations per participant and iteration", "evaluate_fit", counts["evaluate_fit"]),
        ("share of profiled time", "fmin_l_bfgs_b", inclusive(stats, "_lbfgsb_py.py", "fmin_l_bfgs_b") / total),
        ("share of profiled time", "hessian", (inclusive(stats, "optimisers.py", "numerical_hessian")
                                               + inclusive(stats, "optimisers.py", "finite_difference_hessian")) / total),
        ("share of profiled time", "evaluate_fit", inclusive(stats, "optimisers.py", "evaluate_fit") / total),
        ("share of profiled time", "update_population", inclusive(stats, "variational.py", "update_population") / total),
        ("share of profiled time", "get_lme", inclusive(stats, "variational.py", "get_lme") / total),
        ("share of profiled time", "objective (all callers)", inclusive(stats, "optimisers.py", "objective") / total),
        ("share of objective", "SessionWrapper.run", inclusive_from(stats, objective_key, "session.py", "run")),
        ("share of objective", "  of which the RLRW model function", inclusive(stats, "reinforcement_learning.py", "__call__")),
        ("share of objective", "Wrapper.reset", inclusive_from(stats, objective_key, "session.py", "reset")),
        ("share of objective", "deepcopy", inclusive_from(stats, objective_key, "copy.py", "deepcopy")),
        ("share of objective", "loss", inclusive_from(stats, objective_key, "minimise.py", "bernoulli")),
        ("share of objective", "Parameters.PDF", inclusive_from(stats, objective_key, "parameters.py", "PDF")),
    ]
    objective_seconds = inclusive(stats, "optimisers.py", "objective")
    rows = [(s, i, v / objective_seconds if s == "share of objective" else v) for s, i, v in rows]
    table = pd.DataFrame(rows, columns=["section", "item", "value"])
    table["value"] = table["value"].round(4)
    print(table.to_string(index=False))

    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / f"{args.label}.csv", index=False)


if __name__ == "__main__":
    main()
