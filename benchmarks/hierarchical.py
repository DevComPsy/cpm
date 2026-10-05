"""
How long does a hierarchical analysis take? The two hierarchical tutorials, timed.

    python benchmarks/hierarchical.py ANALYSIS MODEL OUT

ANALYSIS is "eb" (docs/tutorials/hierarchical-empirical-bayes.ipynb: separate
maximum-likelihood fits, then empirical Bayes) or "vb"
(docs/tutorials/hierarchical-variational-bayes.ipynb: variational Bayes). Both
simulate 50 participants of the bandit task from a known group distribution
and fit them with `FminBound` (2 starts per participant), exactly as the
tutorials do, with the same random seed.

MODEL is "tutorial" (the per-trial model the tutorials build from `Softmax`
and `SeparableRule`, in a `Wrapper`) or "rlrw" (the same model as the built-in
`cpm.applications.reinforcement_learning.RLRW`, which computes all trials at
once and is compiled with numba if it is installed). The data are simulated
with the tutorial model in both cases, so both fit the same data.

Writes the timings and the results (estimates and hyperparameters) to OUT, a
pickle file.
"""

import pickle
import platform
import sys
import time
import warnings
from functools import partial

import numpy as np
import pandas as pd

import cpm
from cpm.datasets import load_bandit_data
from cpm.generators import Parameters, Simulator, Value, Wrapper
from cpm.models.decision import Softmax
from cpm.models.learning import SeparableRule
from cpm.optimisation import FminBound, minimise

ANALYSIS, MODEL, OUT = sys.argv[1:4]
assert ANALYSIS in ("eb", "vb") and MODEL in ("tutorial", "rlrw")

np.random.seed(2026)
_ = np.seterr(all="ignore")
warnings.simplefilter("ignore")


## ---------------------------------------------------------------------------
## the tutorials' model and data (cell 3 of both notebooks)


def model(parameters, trial, generate=False):
    values = np.asarray(parameters.values).copy()
    stimuli = np.array([trial.arm_left, trial.arm_right]).astype(int)
    rewards = np.array([trial.reward_left, trial.reward_right])

    choice_rule = Softmax(activations=values[stimuli - 1], temperature=parameters.temperature)
    choice_rule.compute()
    choice = choice_rule.choice() if generate else int(trial.response)

    chosen = np.zeros(4)
    chosen[stimuli[choice] - 1] = 1
    update = SeparableRule(weights=values, feedback=[rewards[choice]], input=chosen, alpha=parameters.alpha)
    update.compute()
    values += update.weights.flatten()

    return {
        "policy": choice_rule.policies,
        "response": choice,
        "values": values,
        "dependent": np.array([choice_rule.policies[1]]),
    }


def make_parameters(alpha_mean, alpha_sd, temperature_mean, temperature_sd):
    return Parameters(
        alpha=Value(value=0.5, lower=1e-10, upper=1, prior="truncated_normal", args={"mean": alpha_mean, "sd": alpha_sd}),
        temperature=Value(value=1, lower=0, upper=10, prior="truncated_normal", args={"mean": temperature_mean, "sd": temperature_sd}),
        values=np.array([0.25, 0.25, 0.25, 0.25]),
    )


timings = {}
start = time.perf_counter()
data = load_bandit_data()
data = data[data.ppt <= 50].reset_index(drop=True)
data["observed"] = data["response"]
population = make_parameters(alpha_mean=0.6, alpha_sd=0.2, temperature_mean=2, temperature_sd=1)
true_parameters = pd.DataFrame(population.sample(size=data.ppt.nunique()))
simulator = Simulator(
    wrapper=Wrapper(model=partial(model, generate=True), parameters=population, data=data[data.ppt == 1]),
    parameters=true_parameters,
    data=data.groupby("ppt"),
)
simulator.run()
data["response"] = simulator.export()["response"].to_numpy()
data["observed"] = data["response"]
timings["simulate"] = time.perf_counter() - start


def make_wrapper():
    """The model with the broad starting priors of the tutorials."""
    if MODEL == "tutorial":
        parameters = make_parameters(alpha_mean=0.5, alpha_sd=0.5, temperature_mean=5, temperature_sd=5)
        return Wrapper(model=model, parameters=parameters, data=data[data.ppt == 1])
    from cpm.applications.reinforcement_learning import RLRW

    wrapper = RLRW(data=data[data.ppt == 1], dimensions=4, parameters_settings=[[0.5, 1e-10, 1], [1, 0, 10]])
    wrapper.parameters.update_prior(alpha={"mean": 0.5, "sd": 0.5}, temperature={"mean": 5, "sd": 5})
    return wrapper


def optimiser(prior):
    return FminBound(
        model=make_wrapper(),
        data=data.groupby("ppt"),
        minimisation=minimise.LogLikelihood.bernoulli,
        prior=prior,
        number_of_starts=2,
        ppt_identifier="ppt",
        parallel=False,
        display=False,
        approx_grad=True,
    )


results = {"true_parameters": true_parameters}
if ANALYSIS == "eb":
    from cpm.hierarchical import EmpiricalBayes

    start = time.perf_counter()
    separate = optimiser(prior=False)
    separate.optimise()
    timings["separate fits"] = time.perf_counter() - start
    results["maximum_likelihood"] = pd.DataFrame(separate.parameters)

    start = time.perf_counter()
    eb = EmpiricalBayes(optimiser=optimiser(prior=True), iteration=6, chain=2, tolerance=1e-3, quiet=True)
    eb.optimise()
    timings["empirical Bayes"] = time.perf_counter() - start
    results["hyperparameters"] = eb.hyperparameters
    results["fit"] = eb.fit
else:
    from cpm.hierarchical import VariationalBayes

    start = time.perf_counter()
    number_of_parameters = 2
    vb = VariationalBayes(
        optimiser=optimiser(prior=True),
        iteration=6,
        chain=2,
        convergence="lme",
        tolerance_lme=1e-2,
        hyperpriors={
            "a0": np.zeros(number_of_parameters),
            "b": 1,
            "v": 0.5,
            "s": np.repeat(0.01, number_of_parameters),
        },
        quiet=True,
    )
    vb.optimise()
    timings["variational Bayes"] = time.perf_counter() - start
    results["hyperparameters"] = vb.hyperparameters
    results["fit"] = vb.fit

try:
    from cpm.core import _jit

    numba_used = MODEL == "rlrw" and bool(_jit.JIT_ENABLED)
except ImportError:
    numba_used = False
record = {
    "analysis": ANALYSIS,
    "model": MODEL,
    "numba": numba_used,
    "cpm": cpm.__version__,
    "python": platform.python_version(),
    "numpy": np.__version__,
    "timings": timings,
    "results": results,
}
with open(OUT, "wb") as f:
    pickle.dump(record, f)
print(f"{ANALYSIS} {MODEL} (cpm {cpm.__version__}, numba {'on' if numba_used else 'off'}): "
      + ", ".join(f"{k} {v:.1f} s" for k, v in timings.items()))
