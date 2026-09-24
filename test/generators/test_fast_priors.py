import copy
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cpm.generators import Parameters, Value, Wrapper
from cpm.generators._fast_priors import fast_logpdf

## the priors Value builds from a string, as (prior, args, lower, upper)
BUILT_IN = [
    ("uniform", None, 0.1, 3.0),
    ("truncated_normal", {"mean": 0.5, "sd": 0.25}, 0.0, 1.0),
    ("truncated_normal", {"mean": 5, "sd": 2.5}, 0.0, 10.0),
    ("truncated_normal", {"mean": 0.0, "sd": 1.0}, -5.0, 5.0),
    ("beta", {"a": 2.0, "b": 3.0, "mean": 0.0, "sd": 1.0}, 0.0, 1.0),
    ("gamma", {"a": 2.5, "mean": 0.0, "sd": 1.5}, 0.0, 20.0),
    ("gamma", {"a": 0.7, "mean": 0.0, "sd": 1.0}, 0.0, 20.0),
    ("truncated_exponential", {"mean": 0.0, "sd": 2.0}, 0.0, 10.0),
    ("norm", {"mean": 1.0, "sd": 2.0}, -10.0, 10.0),
]


@pytest.mark.parametrize("prior, args, lower, upper", BUILT_IN)
def test_logpdf_equals_scipy_inside_and_outside_the_support(prior, args, lower, upper):
    value = Value(value=lower, lower=lower, upper=upper, prior=prior, args=args)
    rng = np.random.default_rng(1)
    points = np.concatenate(
        [
            rng.uniform(lower - 1, upper + 1, 200),
            [lower, upper, lower - 1e-9, upper + 1e-9, (lower + upper) / 2],
        ]
    )
    for x in points:
        value.fill(x)
        expected = value.prior.logpdf(x)
        got = value.PDF(log=True)
        if np.isfinite(expected):
            assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)
        else:
            assert got == expected
        assert isinstance(got, np.float64)


@pytest.mark.parametrize(
    "frozen",
    [
        stats.norm(0.3, 2.0),  # positional arguments
        stats.gamma(2.0, scale=3.0),
        stats.beta(0.5, 0.5),
        stats.truncnorm(-1.0, 3.0, loc=1.0, scale=0.5),
        stats.uniform(-2, 4),
    ],
)
def test_logpdf_of_user_supplied_frozen_distributions(frozen):
    for x in np.linspace(-3, 6, 101):
        expected = frozen.logpdf(x)
        got = fast_logpdf(frozen, x)
        got = frozen.logpdf(x) if got is None else got
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-12) or got == expected


def test_other_priors_and_values_fall_back_to_scipy():
    ## unsupported family
    assert fast_logpdf(stats.lognorm(0.5), 1.0) is None
    ## array-valued points
    assert fast_logpdf(stats.norm(0, 1), np.array([0.1, 0.2])) is None
    ## objects that are not scipy distributions
    assert fast_logpdf(object(), 0.1) is None
    ## non-finite points
    assert fast_logpdf(stats.norm(0, 1), np.nan) is None
    mvn = Value(
        value=np.zeros(2),
        lower=-1,
        upper=1,
        prior=stats.multivariate_normal,
        args={"mean": np.zeros(2), "cov": np.eye(2)},
    )
    assert mvn.PDF(log=True) == pytest.approx(
        stats.multivariate_normal(np.zeros(2), np.eye(2)).logpdf(np.zeros(2))
    )


def test_priors_changed_in_place_are_picked_up():
    value = Value(value=0.4, lower=0, upper=1, prior="truncated_normal",
                  args={"mean": 0.5, "sd": 0.25})
    before = value.PDF(log=True)
    value.prior.kwds.update(loc=0.9, scale=0.05, a=(0 - 0.9) / 0.05, b=(1 - 0.9) / 0.05)
    after = value.PDF(log=True)
    assert after != before
    assert after == pytest.approx(value.prior.logpdf(0.4), rel=1e-12)


def test_copies_share_the_prior_but_not_the_value():
    value = Value(value=np.array([0.1, 0.2]), lower=0, upper=1, prior="uniform")
    copied = copy.deepcopy(value)
    assert copied.prior is value.prior
    assert copied.__pdef__ == "uniform"
    copied.value[0] = 0.9
    assert value.value[0] == 0.1


def test_update_prior_does_not_leak_into_copies():
    parameters = Parameters(
        a=Value(value=0.5, lower=0, upper=1, prior="truncated_normal",
                args={"mean": 0.5, "sd": 0.25})
    )
    copied = copy.deepcopy(parameters)
    original = copied.PDF(log=True)
    parameters.update_prior(a={"mean": 0.9, "sd": 0.05})
    assert parameters.a.prior is not copied.a.prior
    assert copied.PDF(log=True) == original
    assert parameters.PDF(log=True) != original
    assert isinstance(
        parameters.a.prior, stats._distn_infrastructure.rv_continuous_frozen
    )


def _wrapper():
    parameters = Parameters(
        a=Value(value=0.5, lower=0, upper=1, prior="truncated_normal",
                args={"mean": 0.5, "sd": 0.25}),
        state=np.zeros(2),
    )

    def model(parameters, trial):
        state = np.asarray(parameters.state) + parameters.a.value
        return {"state": state, "dependent": np.array([parameters.a.value])}

    data = pd.DataFrame({"x": np.zeros(3), "observed": np.zeros(3)})
    return Wrapper(model=model, data=data, parameters=parameters)


def test_reset_restores_values_and_states():
    wrapper = _wrapper()
    wrapper.reset(parameters=[0.2])
    wrapper.run()
    assert np.allclose(wrapper.parameters.state.value, 0.6)
    wrapper.reset()
    assert wrapper.parameters.a.value == 0.5
    assert np.allclose(wrapper.parameters.state.value, 0.0)
    assert np.allclose(wrapper.__init_parameters__.state.value, 0.0)


def test_reset_keeps_updated_priors():
    """cpm.hierarchical updates the priors of model.parameters between fits."""
    wrapper = _wrapper()
    wrapper.parameters.update_prior(a={"mean": 0.9, "sd": 0.05})
    updated = wrapper.parameters.PDF(log=True)
    for _ in range(3):
        wrapper.reset(parameters=[0.5])
        wrapper.run()
    wrapper.reset(parameters=[0.5])
    assert wrapper.parameters.PDF(log=True) == updated


def test_hierarchical_prior_updates_reach_the_objective():
    """The prior that update_prior sets is the one the objective uses."""
    from cpm.core.optimisers import objective
    from cpm.optimisation.minimise import LogLikelihood

    warnings.simplefilter("ignore")
    wrapper = _wrapper()
    observed = np.zeros((3, 1))
    first = objective([0.5], wrapper, observed, LogLikelihood.bernoulli, True)
    wrapper.parameters.update_prior(a={"mean": 0.9, "sd": 0.05})
    values = [objective([0.5], wrapper, observed, LogLikelihood.bernoulli, True)
              for _ in range(3)]
    assert values[0] != first
    assert values[0] == values[1] == values[2]
