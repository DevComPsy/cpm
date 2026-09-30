"""Optimisers accept arbitrarily many user-supplied goodness-of-fit metrics and
record them alongside the fit, so that they reach the exported frame.

Metrics follow the convention already used by
`cpm.optimisation.compare.PenalisedLikelihoods` and the loss functions in
`cpm.optimisation.minimise`: they are called with keyword arguments only and
absorb anything they do not need in `**kwargs`.
"""

import io
import contextlib

import numpy as np
import pandas as pd
import pytest

from cpm.core.optimisers import evaluate_fit, fit_extras, normalise_metrics
from cpm.generators import Parameters, Value, Wrapper
from cpm.optimisation import (
    DifferentialEvolution,
    Fmin,
    FminBound,
    Minimize,
    minimise,
)
from cpm.optimisation.compare import PenalisedLikelihoods

try:
    from cpm.optimisation import Bads

    HAS_BADS = True
except ImportError:  # pragma: no cover - pybads is an optional dependency
    HAS_BADS = False


PARTICIPANTS = (1, 2)
N_TRIALS = 40
N_PARAMETERS = 2
METRICS = [PenalisedLikelihoods.BIC, PenalisedLikelihoods.AIC, minimise.Distance.SSE]


def saturating_model(parameters, trial):
    return {
        "dependent": np.array(
            [parameters.alpha * np.tanh(parameters.beta * trial["stimulus"])]
        )
    }


def make_data(seed=7):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "ppt": np.repeat(PARTICIPANTS, N_TRIALS),
            "stimulus": rng.uniform(-2, 2, len(PARTICIPANTS) * N_TRIALS),
        }
    )
    data["observed"] = 0.7 * np.tanh(1.5 * data.stimulus) + rng.normal(
        0, 0.05, len(data)
    )
    return data


def make_parameters():
    return Parameters(
        alpha=Value(
            value=0.5, lower=0.05, upper=2.0, prior="norm", args={"mean": 0.7, "sd": 0.5}
        ),
        beta=Value(
            value=1.0, lower=0.05, upper=3.0, prior="norm", args={"mean": 1.5, "sd": 0.5}
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


def fit_with(name, prior, metrics=METRICS):
    optimiser_class, extra = OPTIMISERS[name]
    data = make_data()
    optimiser = optimiser_class(
        model=make_wrapper(data),
        data=data,
        minimisation=minimise.LogLikelihood.continuous,
        ppt_identifier="ppt",
        prior=prior,
        display=False,
        metrics=metrics,
        **extra,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        optimiser.optimise()
    return optimiser


@pytest.fixture(scope="module")
def fitted():
    """Each optimiser fitted once with and once without a prior, shared by the
    tests below."""
    return {
        (name, prior): fit_with(name, prior)
        for name in NAMES
        for prior in (True, False)
    }


class TestNormaliseMetrics:
    def test_none_means_no_metrics(self):
        assert normalise_metrics(None) == {}

    def test_a_single_callable_is_accepted(self):
        assert normalise_metrics(PenalisedLikelihoods.AIC) == {
            "AIC": PenalisedLikelihoods.AIC
        }

    def test_an_iterable_is_named_after_its_callables(self):
        assert list(normalise_metrics(METRICS)) == ["BIC", "AIC", "SSE"]

    def test_a_mapping_keeps_the_names_it_was_given(self):
        metrics = {"bayesian": PenalisedLikelihoods.BIC}
        assert list(normalise_metrics(metrics)) == ["bayesian"]

    def test_a_lambda_can_be_named_through_a_mapping(self):
        """`__name__` is `<lambda>` for every lambda, so a mapping is the only way
        to give more than one of them a usable column name."""
        metrics = normalise_metrics({"half": lambda likelihood, **kwargs: likelihood / 2})
        assert metrics["half"](likelihood=8.0) == 4.0

    def test_something_uncallable_is_rejected_immediately(self):
        """Rejecting at construction beats failing mid-fit, when the traceback no
        longer points at the call that caused it."""
        with pytest.raises(TypeError, match="not callable"):
            normalise_metrics({"BIC": 3.0})


class TestFitExtras:
    def test_it_lists_the_prior_terms_before_the_metrics(self):
        assert fit_extras(prior=True, metrics=METRICS) == [
            "log_likelihood",
            "log_prior",
            "BIC",
            "AIC",
            "SSE",
        ]

    def test_without_a_prior_only_the_metrics_are_listed(self):
        assert fit_extras(prior=False, metrics=METRICS) == ["BIC", "AIC", "SSE"]

    def test_nothing_is_recorded_by_default(self):
        assert fit_extras() == []


class TestEvaluateFit:
    @pytest.fixture
    def problem(self):
        data = make_data()
        wrapper = make_wrapper(data)
        observed = data[data.ppt == PARTICIPANTS[0]].observed.to_numpy()
        return wrapper, observed, minimise.LogLikelihood.continuous

    def test_metrics_receive_the_documented_arguments(self, problem):
        """The docstring promises a specific set of keyword arguments; a metric
        written against it must not break because one quietly disappears."""
        wrapper, observed, loss = problem
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            return 0.0

        evaluate_fit(
            np.array([0.7, 1.5]), wrapper, observed, loss, prior=True, metrics={"spy": spy}
        )
        assert set(seen) == {
            "likelihood",
            "log_likelihood",
            "log_prior",
            "predicted",
            "observed",
            "n",
            "k",
            "parameters",
        }
        assert seen["n"] == N_TRIALS
        assert seen["k"] == N_PARAMETERS
        assert seen["likelihood"] == seen["log_likelihood"]

    def test_the_likelihood_handed_over_excludes_the_prior(self, problem):
        """BIC and AIC are defined on the likelihood. Passing the log posterior
        instead would silently bias every model comparison."""
        wrapper, observed, loss = problem
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            return 0.0

        output = evaluate_fit(
            np.array([0.7, 1.5]), wrapper, observed, loss, prior=True, metrics={"spy": spy}
        )
        assert seen["likelihood"] == output["log_likelihood"]
        assert seen["likelihood"] != output["log_likelihood"] + output["log_prior"]

    def test_a_metric_may_not_shadow_a_recorded_quantity(self, problem):
        wrapper, observed, loss = problem
        with pytest.raises(ValueError, match="log_prior"):
            evaluate_fit(
                np.array([0.7, 1.5]),
                wrapper,
                observed,
                loss,
                prior=True,
                metrics={"log_prior": lambda **kwargs: 0.0},
            )

    def test_no_metrics_means_no_extra_entries(self, problem):
        wrapper, observed, loss = problem
        assert evaluate_fit(np.array([0.7, 1.5]), wrapper, observed, loss) == {}


class TestOptimiserOutput:
    @pytest.mark.parametrize("prior", [True, False], ids=["prior", "no-prior"])
    @pytest.mark.parametrize("name", NAMES)
    def test_every_metric_is_recorded(self, fitted, name, prior):
        for fit in fitted[(name, prior)].fit:
            for metric in ("BIC", "AIC", "SSE"):
                assert np.isfinite(fit[metric])

    @pytest.mark.parametrize("prior", [True, False], ids=["prior", "no-prior"])
    @pytest.mark.parametrize("name", NAMES)
    def test_every_metric_reaches_the_exported_frame(self, fitted, name, prior):
        exported = fitted[(name, prior)].export()
        for metric in ("BIC", "AIC", "SSE"):
            assert metric in exported.columns
        assert len(exported) == len(PARTICIPANTS)

    @pytest.mark.parametrize("name", NAMES)
    def test_the_recorded_values_match_the_definitions(self, fitted, name):
        """Recomputed by hand from the recorded log likelihood, so a metric
        receiving the wrong `likelihood`, `n` or `k` shows up here."""
        for fit in fitted[(name, True)].fit:
            likelihood = fit["log_likelihood"]
            assert np.isclose(
                fit["AIC"], -2 * likelihood + 2 * N_PARAMETERS, atol=1e-9
            )
            assert np.isclose(
                fit["BIC"],
                -2 * likelihood + N_PARAMETERS * np.log(N_TRIALS),
                atol=1e-9,
            )

    @pytest.mark.parametrize("name", NAMES)
    def test_penalised_likelihoods_ignore_the_prior(self, fitted, name):
        """With a prior, `fun` is the negative log posterior. A metric built on
        `-fun` rather than the likelihood would come out different, and wrong."""
        for fit in fitted[(name, True)].fit:
            assert not np.isclose(fit["AIC"], -2 * -fit["fun"] + 2 * N_PARAMETERS)

    @pytest.mark.parametrize("name", NAMES)
    def test_a_custom_metric_is_recorded_under_its_mapping_name(self, name):
        optimiser = fit_with(
            name, prior=False, metrics={"trials": lambda observed, **kwargs: len(observed)}
        )
        for fit in optimiser.fit:
            assert fit["trials"] == N_TRIALS
        assert "trials" in optimiser.export().columns

    @pytest.mark.parametrize("name", NAMES)
    def test_no_metrics_leaves_the_output_untouched(self, name):
        optimiser = fit_with(name, prior=False, metrics=None)
        for fit in optimiser.fit:
            for metric in ("BIC", "AIC", "SSE"):
                assert metric not in fit
