import numpy as np
import pandas as pd
import pytest

from cpm.generators import Parameters, Value, Wrapper
from cpm.optimisation import Fmin, FminBound, Minimize, minimise

try:
    from cpm.optimisation import Bads

    HAS_BADS = True
except ImportError:  # pragma: no cover - pybads is an optional dependency
    HAS_BADS = False


def simple_model(parameters, trial):
    """A one-parameter model, deliberately trivial: these tests are about the
    optimiser's bookkeeping, not about fitting quality."""
    return {"dependent": np.array([trial["stimulus"] * parameters.alpha])}


def make_data(n_ppt=2, n_trials=3):
    return pd.DataFrame(
        {
            "ppt": np.repeat(np.arange(1, n_ppt + 1), n_trials),
            "stimulus": np.tile(np.arange(1, n_trials + 1), n_ppt) * 1.0,
            "observed": np.tile(np.linspace(0.1, 0.3, n_trials), n_ppt),
        }
    )


def make_wrapper(data):
    parameters = Parameters(
        alpha=Value(
            value=0.5,
            lower=0.0,
            upper=1.0,
            prior="norm",
            args={"mean": 0.5, "sd": 0.25},
        )
    )
    return Wrapper(
        model=simple_model,
        data=data[data.ppt == 1],
        parameters=parameters,
    )


def build(optimiser_class, **kwargs):
    data = make_data()
    defaults = dict(
        model=make_wrapper(data),
        data=data,
        minimisation=minimise.LogLikelihood.continuous,
        ppt_identifier="ppt",
    )
    defaults.update(kwargs)
    return optimiser_class(**defaults)


OPTIMISERS = [Fmin, FminBound, Minimize]
if HAS_BADS:
    OPTIMISERS.append(Bads)

IDS = [cls.__name__ for cls in OPTIMISERS]


class TestInitialGuessProvenance:
    """`number_of_starts` and the origin of `initial_guess` must survive
    construction.

    Before this, `number_of_starts` was consumed in `__init__` and discarded,
    recoverable only as `initial_guess.shape[0]`, and there was no record at all
    of whether the guesses were supplied by the user or drawn at random. With
    `number_of_starts > 1` and no explicit guess, the same data, bounds and
    estimator can converge to different optima across runs, and nothing on the
    fitted object said so.
    """

    @pytest.mark.parametrize("optimiser_class", OPTIMISERS, ids=IDS)
    def test_number_of_starts_is_stored(self, optimiser_class):
        optimiser = build(optimiser_class, number_of_starts=3)
        assert optimiser.number_of_starts == 3
        # And it still agrees with the array it generated.
        assert optimiser.initial_guess.shape[0] == 3

    @pytest.mark.parametrize("optimiser_class", OPTIMISERS, ids=IDS)
    def test_default_number_of_starts_is_stored(self, optimiser_class):
        optimiser = build(optimiser_class)
        assert optimiser.number_of_starts == 1

    @pytest.mark.parametrize("optimiser_class", OPTIMISERS, ids=IDS)
    def test_drawn_guesses_are_flagged_as_not_supplied(self, optimiser_class):
        optimiser = build(optimiser_class, number_of_starts=2)
        assert optimiser.initial_guess_supplied is False

    @pytest.mark.parametrize("optimiser_class", OPTIMISERS, ids=IDS)
    def test_supplied_guesses_are_flagged_as_supplied(self, optimiser_class):
        guesses = [[0.2], [0.8]]
        optimiser = build(optimiser_class, number_of_starts=2, initial_guess=guesses)
        assert optimiser.initial_guess_supplied is True
        assert np.allclose(optimiser.initial_guess, np.asarray(guesses))

    @pytest.mark.parametrize("optimiser_class", OPTIMISERS, ids=IDS)
    def test_supplied_and_drawn_are_distinguishable(self, optimiser_class):
        """The distinction the attribute exists for."""
        drawn = build(optimiser_class, number_of_starts=2)
        supplied = build(
            optimiser_class, number_of_starts=2, initial_guess=[[0.2], [0.8]]
        )
        assert drawn.number_of_starts == supplied.number_of_starts
        assert drawn.initial_guess_supplied != supplied.initial_guess_supplied


class TestResetKeepsProvenanceHonest:
    @pytest.mark.parametrize("optimiser_class", OPTIMISERS, ids=IDS)
    def test_reset_preserves_number_of_starts(self, optimiser_class):
        optimiser = build(optimiser_class, number_of_starts=3)
        optimiser.reset()
        assert optimiser.number_of_starts == 3
        assert optimiser.initial_guess.shape[0] == 3

    @pytest.mark.parametrize("optimiser_class", OPTIMISERS, ids=IDS)
    def test_reset_clears_the_supplied_flag(self, optimiser_class):
        """`reset(initial_guess=True)` draws new random guesses, so guesses that
        were supplied are gone. The flag describes the array, not the argument
        passed to `__init__`, so it has to follow."""
        optimiser = build(
            optimiser_class, number_of_starts=2, initial_guess=[[0.2], [0.8]]
        )
        assert optimiser.initial_guess_supplied is True

        optimiser.reset()
        assert optimiser.initial_guess_supplied is False
        assert optimiser.number_of_starts == 2

    @pytest.mark.parametrize("optimiser_class", OPTIMISERS, ids=IDS)
    def test_reset_without_regenerating_keeps_the_flag(self, optimiser_class):
        """`reset(initial_guess=False)` leaves the guesses alone, so a supplied
        set is still supplied afterwards."""
        guesses = [[0.2], [0.8]]
        optimiser = build(optimiser_class, number_of_starts=2, initial_guess=guesses)
        optimiser.reset(initial_guess=False)
        assert optimiser.initial_guess_supplied is True
        assert np.allclose(optimiser.initial_guess, np.asarray(guesses))


class TestProvenanceSurvivesFitting:
    def test_attributes_are_intact_after_optimise(self):
        """A fitted optimiser must still be able to answer how it was started.

        Only `FminBound` is exercised here: running a real fit is slow, and the
        attributes are set in `__init__` for every optimiser, which the
        parametrised tests above already cover.
        """
        optimiser = build(
            FminBound,
            number_of_starts=2,
            approx_grad=True,
        )
        optimiser.optimise()

        assert optimiser.number_of_starts == 2
        assert optimiser.initial_guess_supplied is False
        assert len(optimiser.fit) > 0
