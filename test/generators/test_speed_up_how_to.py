"""
The example of docs/how-to/speed-up-your-model.md: the model of Tutorial 1, written
for all trials at once in a SessionWrapper, gives the same results as trial by trial.

The code blocks of the page are run as they are, from a file (numba can only cache
functions defined in a file), so that the page and this test cannot drift apart.
"""

import importlib.util
import pathlib
import re
import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest

from cpm.core.optimisers import objective
from cpm.generators import SessionWrapper, Simulator, Wrapper
from cpm.optimisation import minimise

PAGE = pathlib.Path(__file__).parents[2] / "docs" / "how-to" / "speed-up-your-model.md"
HAVE_NUMBA = importlib.util.find_spec("numba") is not None

pytestmark = pytest.mark.skipif(not PAGE.exists(), reason="the documentation is not available")


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    blocks = re.findall(r"```python\n(.*?)```", PAGE.read_text(encoding="utf-8"), flags=re.S)
    if not HAVE_NUMBA:
        blocks = [block for block in blocks if "njit" not in block]
    path = tmp_path_factory.mktemp("how_to") / "speed_up_your_model.py"
    path.write_text("\n".join(blocks), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("speed_up_your_model", path)
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(module)
    return module


def test_the_check_on_the_page_passes(page):
    ## the check of the page, with exact equality instead of np.allclose
    for alpha, temperature in [(0.1, 1.0), (0.5, 5.0), (0.9, 9.0)]:
        for wrapper in (page.per_trial, page.all_trials):
            wrapper.reset(parameters={"alpha": alpha, "temperature": temperature})
            wrapper.run()
        np.testing.assert_array_equal(page.all_trials.dependent, page.per_trial.dependent)


def test_the_objective_and_export_are_the_same(page):
    observed = page.one.observed.to_numpy()
    loss = minimise.LogLikelihood.bernoulli
    rng = np.random.default_rng(1)
    points = [[rng.uniform(1e-10, 1), rng.uniform(0, 10)] for _ in range(20)]
    versions = [page.all_trials] + ([page.compiled] if HAVE_NUMBA else [])
    expected = [objective(x, page.per_trial, observed, loss, True) for x in points]
    for wrapper in versions:
        assert [objective(x, wrapper, observed, loss, True) for x in points] == pytest.approx(
            expected, rel=1e-12, abs=1e-12
        )
    for wrapper in (page.per_trial, page.all_trials):
        wrapper.reset()
        wrapper.run()
    pd.testing.assert_frame_equal(page.all_trials.export(), page.per_trial.export(), check_exact=False, rtol=0, atol=1e-12)


def test_simulations_make_the_same_choices(page):
    warnings.simplefilter("ignore")
    data = page.experiment[page.experiment.ppt <= 2].groupby("ppt")
    draws = pd.DataFrame({"alpha": [0.2, 0.6], "temperature": [2.0, 5.0]})
    exports = []
    for function, wrapper in ((page.model, Wrapper), (page.session_model, SessionWrapper)):
        np.random.seed(5)
        simulator = Simulator(
            wrapper=wrapper(model=partial(function, generate=True), parameters=page.parameters, data=page.one),
            parameters=draws,
            data=data,
        )
        simulator.run()
        exports.append(simulator.export())
    pd.testing.assert_frame_equal(exports[1], exports[0], check_exact=False, rtol=0, atol=1e-12)
