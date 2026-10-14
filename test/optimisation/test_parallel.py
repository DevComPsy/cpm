"""
`parallel=True` sends the model to the workers with dill. dill copied the classes
of `cpm.generators` by value instead of by reference, because the package
deleted its submodules, so `isinstance` checks failed in the workers: after the
first run, `Wrapper.reset` replaced the priors updated with `update_prior` with
the priors the model was created with, and fits with `prior=True` differed from
serial ones.
"""

import dill
import numpy as np
import pytest

from cpm.generators import LogParameters, Parameters, SessionWrapper, Simulator, Value, Wrapper
from cpm.optimisation import FminBound

from test.optimisation.test_fminbound_gradient import LOSS, bandit, rlrw


@pytest.mark.parametrize("cls", [Value, Parameters, LogParameters, Wrapper, SessionWrapper, Simulator])
def test_dill_pickles_the_classes_by_reference(cls):
    assert dill.loads(dill.dumps(cls)) is cls


def test_a_copy_keeps_updated_priors_across_runs():
    data = bandit(participants=1)
    model = dill.loads(dill.dumps(rlrw(data)))
    for x in ([0.3, 3.0], [0.05, 1.0]):
        model.reset(parameters=np.array(x))
        model.run()
        assert model.parameters.alpha.prior.kwds["scale"] == 0.5


def fit(parallel):
    data = bandit(participants=4, trials=60, seed=3)
    optimiser = FminBound(
        model=rlrw(data),
        data=data.groupby("ppt"),
        initial_guess=[[0.3, 3.0]],
        minimisation=LOSS,
        prior=True,
        approx_grad=True,
        parallel=parallel,
        cl=2,
        ppt_identifier="ppt",
    )
    optimiser.optimise()
    return optimiser


def test_parallel_fits_with_updated_priors_equal_serial_ones():
    serial, parallel = fit(False), fit(True)
    for a, b in zip(serial.fit, parallel.fit):
        np.testing.assert_array_equal(a["x"], b["x"])
        assert a["fun"] == b["fun"]
