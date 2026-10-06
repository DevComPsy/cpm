"""`start="prior_mean"`: the first start of every participant is the location of the current prior."""

import warnings

import numpy as np
import pytest

from cpm.applications.reinforcement_learning import RLRW
from cpm.datasets import load_bandit_data
from cpm.hierarchical import EmpiricalBayes, VariationalBayes
from cpm.optimisation import FminBound, minimise

METHODS = [EmpiricalBayes, VariationalBayes]


def method(cls, participants=4, iterations=2, chain=1, **kwargs):
    warnings.simplefilter("ignore")
    data = load_bandit_data()
    data["observed"] = data["response"]
    data = data[data.ppt.isin(sorted(data.ppt.unique())[:participants])].copy()
    model = RLRW(data=data[data.ppt == data.ppt.iloc[0]], dimensions=4, parameters_settings=[[0.5, 1e-2, 1.0], [2.0, 0.0, 10.0]])
    model.parameters.update_prior(alpha={"mean": 0.3, "sd": 0.5}, temperature={"mean": 12.0, "sd": 5.0})
    optimiser = FminBound(
        model=model,
        data=data,
        minimisation=minimise.LogLikelihood.bernoulli,
        prior=True,
        ppt_identifier="ppt",
        number_of_starts=2,
        approx_grad=True,
    )
    tolerance = {"tolerance": 0.0} if cls is EmpiricalBayes else {"tolerance_lme": 0.0, "tolerance_param": 0.0}
    return cls(optimiser=optimiser, iteration=iterations, chain=chain, quiet=True, **tolerance, **kwargs)


def record_guesses(fit):
    """The guesses and the prior locations the optimiser starts from, on every participant-wise step."""
    guesses, locations = [], []
    optimise = fit.optimiser.optimise

    def recorded(*args, **kwargs):
        guesses.append(fit.optimiser.initial_guess.copy())
        parameters = fit.optimiser.model.parameters
        locations.append([getattr(parameters, name).prior.kwds["loc"] for name in parameters.free()])
        return optimise(*args, **kwargs)

    fit.optimiser.optimise = recorded
    return guesses, locations


@pytest.mark.parametrize("cls", METHODS)
def test_first_start_is_the_prior_location_clipped_into_the_bounds(cls):
    fit = method(cls, iterations=3, chain=2, start="prior_mean")
    guesses, locations = record_guesses(fit)
    np.random.seed(1)
    fit.optimise()
    assert len(guesses) == 6
    lower, upper = fit.optimiser.model.parameters.bounds()
    np.testing.assert_array_equal(guesses[0][0], [0.3, 10.0])  # 12 is above the upper bound of 10
    for guess, location in zip(guesses, locations):
        np.testing.assert_array_equal(guess[0], np.clip(location, lower, upper))


@pytest.mark.parametrize("cls", METHODS)
def test_other_starts_are_the_random_ones(cls):
    random, prior_mean = method(cls, start="random"), method(cls, start="prior_mean")
    random_guesses, _ = record_guesses(random)
    prior_guesses, _ = record_guesses(prior_mean)
    np.random.seed(1)
    random.optimise()
    np.random.seed(1)
    prior_mean.optimise()
    np.testing.assert_array_equal(random_guesses[0][1:], prior_guesses[0][1:])
    assert not np.array_equal(random_guesses[0][0], prior_guesses[0][0])


@pytest.mark.parametrize("cls", METHODS)
def test_random_is_the_default(cls):
    default, random = method(cls), method(cls, start="random")
    np.random.seed(1)
    default.optimise()
    np.random.seed(1)
    random.optimise()
    assert default.hyperparameters.equals(random.hyperparameters)
    assert default.fit.equals(random.fit)


@pytest.mark.parametrize("cls", METHODS)
def test_unknown_start_is_an_error(cls):
    with pytest.raises(ValueError, match="start"):
        method(cls, start="previous")
