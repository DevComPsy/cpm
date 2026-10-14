"""
With several starts, every optimiser keeps each participant's best start. Later
starts used to be compared only with the first, so with three or more starts, a
start that was worse than the second but better than the first replaced the
second.
"""

import numpy as np
import pandas as pd
import pytest

from cpm.generators import Parameters, Value, Wrapper
from cpm.optimisation import Fmin, FminBound, Minimize

try:
    from cpm.optimisation import Bads

    HAS_BADS = True
except ImportError:  # pragma: no cover - pybads is an optional dependency
    HAS_BADS = False

## three wells, each start at the bottom of one: -1 at 0.15, -3 at 0.5, -2 at 0.85
CENTRES = np.array([0.15, 0.5, 0.85])
DEPTHS = np.array([1.0, 3.0, 2.0])


def wells(predicted, observed, **kwargs):
    alpha = np.ravel(predicted)[0]
    return -np.sum(DEPTHS * np.exp(-(((alpha - CENTRES) / 0.05) ** 2)))


def identity(parameters, trial):
    return {"dependent": np.array([parameters.alpha.value])}


def build(optimiser_class, **kwargs):
    data = pd.DataFrame({"ppt": [1, 2], "observed": [0.0, 0.0]})
    model = Wrapper(
        model=identity,
        data=data[data.ppt == 1],
        parameters=Parameters(alpha=Value(value=0.5, lower=0.0, upper=1.0, prior="norm", args={"mean": 0.5, "sd": 1.0})),
    )
    return optimiser_class(
        model=model,
        data=data,
        minimisation=wells,
        ppt_identifier="ppt",
        initial_guess=[[0.15], [0.5], [0.85]],
        **kwargs,
    )


CASES = [
    pytest.param(Fmin, {}, id="Fmin"),
    pytest.param(FminBound, {"approx_grad": True}, id="FminBound"),
    pytest.param(Minimize, {}, id="Minimize"),
]
if HAS_BADS:
    CASES.append(
        pytest.param(
            Bads,
            {"options": {"uncertainty_handling": False, "display": "off", "max_fun_evals": 20}},
            id="Bads",
        )
    )


@pytest.mark.parametrize("hessian", ["numdifftools", "finite_differences"])
@pytest.mark.parametrize("optimiser_class, kwargs", CASES)
def test_the_best_of_three_starts_is_kept(optimiser_class, kwargs, hessian):
    optimiser = build(optimiser_class, hessian=hessian, **kwargs)
    optimiser.optimise()
    for fit, parameters in zip(optimiser.fit, optimiser.parameters):
        assert fit["fun"] == pytest.approx(-3.0, abs=1e-3)
        assert parameters["alpha"] == pytest.approx(0.5, abs=1e-2)
