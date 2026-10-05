"""With `prior=True` the value an optimiser minimises is the negative summed log
posterior density, and `fun` alone cannot be turned back into a log likelihood.

Model comparison and goodness-of-fit indices generally need the likelihood and
the prior separately, so every optimiser additionally records `log_likelihood`
and `log_prior` at the optimised parameter values. See
https://github.com/DevComPsy/cpm/issues/28.
"""

import io
import contextlib

import numpy as np
import pandas as pd
import pytest

from cpm.core.optimisers import objective, decompose_objective
from cpm.generators import Parameters, Value, Wrapper
from cpm.optimisation import (
    DifferentialEvolution,
    Fmin,
    FminBound,
    Minimize,
    minimise,
)

try:
    from cpm.optimisation import Bads

    HAS_BADS = True
except ImportError:  # pragma: no cover - pybads is an optional dependency
    HAS_BADS = False


PARTICIPANTS = (1, 2)
TRUE_ALPHA = 0.7
TRUE_BETA = 1.5


def saturating_model(parameters, trial):
    return {
        "dependent": np.array(
            [parameters.alpha * np.tanh(parameters.beta * trial["stimulus"])]
        )
    }


def make_data(n_trials=40, seed=7):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "ppt": np.repeat(PARTICIPANTS, n_trials),
            "stimulus": rng.uniform(-2, 2, len(PARTICIPANTS) * n_trials),
        }
    )
    data["observed"] = TRUE_ALPHA * np.tanh(TRUE_BETA * data.stimulus) + rng.normal(
        0, 0.05, len(data)
    )
    return data


def make_parameters():
    return Parameters(
        alpha=Value(
            value=0.5,
            lower=0.05,
            upper=2.0,
            prior="norm",
            args={"mean": 0.7, "sd": 0.5},
        ),
        beta=Value(
            value=1.0,
            lower=0.05,
            upper=3.0,
            prior="norm",
            args={"mean": 1.5, "sd": 0.5},
        ),
    )


def make_wrapper(data):
    return Wrapper(
        model=saturating_model,
        data=data[data.ppt == PARTICIPANTS[0]],
        parameters=make_parameters(),
    )


OPTIMISERS = {
    "Fmin": (Fmin, {}),
    "FminBound": (FminBound, {"approx_grad": True}),
    "Minimize": (Minimize, {"method": "L-BFGS-B"}),
    "DifferentialEvolution": (
        DifferentialEvolution,
        {"maxiter": 5, "tol": 0.1, "seed": 1},
    ),
}
if HAS_BADS:
    OPTIMISERS["Bads"] = (
        Bads,
        {"options": {"display": "off", "max_fun_evals": 60, "random_seed": 1}},
    )

NAMES = sorted(OPTIMISERS)


def fit_with(name, prior):
    optimiser_class, extra = OPTIMISERS[name]
    data = make_data()
    optimiser = optimiser_class(
        model=make_wrapper(data),
        data=data,
        minimisation=minimise.LogLikelihood.continuous,
        ppt_identifier="ppt",
        prior=prior,
        display=False,
        **extra,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        optimiser.optimise()
    return optimiser


@pytest.fixture(scope="module")
def fitted_with_prior():
    """Every optimiser fitted once with `prior=True`, shared across the tests
    below so that the suite pays for each fit only once."""
    return {name: fit_with(name, prior=True) for name in NAMES}


@pytest.fixture(scope="module")
def fitted_without_prior():
    return {name: fit_with(name, prior=False) for name in NAMES}


class TestDecomposeObjective:
    """`decompose_objective` has to mirror `objective` exactly, or the two parts
    will not add back up to the `fun` an optimiser reports."""

    @pytest.fixture
    def problem(self):
        data = make_data()
        wrapper = make_wrapper(data)
        observed = data[data.ppt == PARTICIPANTS[0]].observed.to_numpy()
        return wrapper, observed, minimise.LogLikelihood.continuous

    @pytest.mark.parametrize("prior", [True, False], ids=["prior", "no-prior"])
    @pytest.mark.parametrize(
        "pars", [[0.7, 1.5], [0.2, 2.8]], ids=["at-optimum", "far-off"]
    )
    def test_the_parts_sum_back_to_the_objective(self, problem, prior, pars):
        wrapper, observed, loss = problem
        pars = np.asarray(pars)
        expected = objective(pars, wrapper, observed, loss, prior)
        log_likelihood, log_prior = decompose_objective(
            pars, wrapper, observed, loss, prior
        )
        assert -(log_likelihood + log_prior) == expected

    def test_without_a_prior_the_log_prior_is_zero(self, problem):
        wrapper, observed, loss = problem
        _, log_prior = decompose_objective(
            np.array([0.7, 1.5]), wrapper, observed, loss, prior=False
        )
        assert log_prior == 0.0

    def test_the_prior_term_is_a_log_density(self, problem):
        """A density below 1 gives a negative log density; the point is that the
        prior term is a log density, not a probability."""
        wrapper, observed, loss = problem
        _, log_prior = decompose_objective(
            np.array([0.7, 1.5]), wrapper, observed, loss, prior=True
        )
        assert log_prior < 0


class TestOptimiserOutput:
    @pytest.mark.parametrize("name", NAMES)
    def test_both_quantities_are_recorded(self, fitted_with_prior, name):
        for fit in fitted_with_prior[name].fit:
            assert np.isfinite(fit["log_likelihood"])
            assert np.isfinite(fit["log_prior"])

    @pytest.mark.parametrize("name", NAMES)
    def test_they_sum_back_to_the_reported_fun(self, fitted_with_prior, name):
        """The whole point: `fun` is the negative log posterior, so the recorded
        parts must reconstruct it rather than merely sit alongside it."""
        for fit in fitted_with_prior[name].fit:
            assert np.isclose(
                fit["fun"],
                -(fit["log_likelihood"] + fit["log_prior"]),
                rtol=0,
                atol=1e-9,
            )

    @pytest.mark.parametrize("name", NAMES)
    def test_the_log_prior_matches_evaluating_the_priors_by_hand(
        self, fitted_with_prior, name
    ):
        """Pins the recorded value to the definition from the issue: update the
        `Parameters` object with the fitted values and take its log PDF.

        Note that `Parameters.update` is keyword-only, so the estimates have
        to be expanded rather than passed as a single dict."""
        optimiser = fitted_with_prior[name]
        parameters = make_parameters()
        for estimates, fit in zip(optimiser.parameters, optimiser.fit):
            parameters.update(**estimates)
            assert np.isclose(fit["log_prior"], parameters.PDF(log=True), atol=1e-9)

    @pytest.mark.parametrize("name", NAMES)
    def test_the_log_likelihood_is_the_fit_without_the_prior(
        self, fitted_with_prior, name
    ):
        """The likelihood must be the data term alone, so it has to differ from
        the posterior whenever the prior contributes anything."""
        for fit in fitted_with_prior[name].fit:
            assert not np.isclose(fit["log_likelihood"], -fit["fun"])

    @pytest.mark.parametrize("name", NAMES)
    def test_nothing_is_recorded_without_a_prior(self, fitted_without_prior, name):
        """Without a prior `fun` is already the negative log likelihood, so the
        keys would be redundant, and a log prior would describe a term that
        never entered the fit."""
        for fit in fitted_without_prior[name].fit:
            assert "log_likelihood" not in fit
            assert "log_prior" not in fit

    @pytest.mark.parametrize("name", NAMES)
    def test_the_quantities_reach_the_exported_frame(self, fitted_with_prior, name):
        exported = fitted_with_prior[name].export()
        assert "log_likelihood" in exported.columns
        assert "log_prior" in exported.columns
        assert len(exported) == len(PARTICIPANTS)
