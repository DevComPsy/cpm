"""`Bads` reaches SciPy indirectly, through pybads and gpyreg, and nothing in
the suite ran `Bads.optimise()` before this file.

That mattered when SciPy 1.18.0 removed the `disp` and `iprint` options of the
L-BFGS-B solver: `FminBound` broke loudly because it calls
`scipy.optimize.fmin_l_bfgs_b` itself, but whether `Bads` was affected could
only be settled by reading `gpyreg` and running a fit by hand. gpyreg fits the
GP hyperparameters with `scipy.optimize.minimize(..., jac=True, bounds=...)`,
which routes to L-BFGS-B, so the same class of change can reach `Bads` - it
simply happened not to this time.

These tests run one real fit and assert that it goes through that SciPy call,
so a future break in the chain fails here instead of in a user's analysis.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.optimize

from cpm.generators import Parameters, Value, Wrapper
from cpm.optimisation import minimise

try:
    from cpm.optimisation import Bads
    from cpm.optimisation.bads import numpy_bool_options
except ImportError:  # pragma: no cover - pybads is an optional dependency
    pytest.skip("pybads is not installed", allow_module_level=True)


TRUE_ALPHA = 0.7
TRUE_BETA = 1.5
INITIAL_GUESS = [[0.5, 1.0]]
PARTICIPANTS = (1, 2)


def saturating_model(parameters, trial):
    """Non-linear in both parameters, so BADS has to search rather than walk
    straight down a gradient."""
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


def make_wrapper(data):
    parameters = Parameters(
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
    return Wrapper(model=saturating_model, data=data[data.ppt == 1], parameters=parameters)


@pytest.fixture(scope="module")
def fitted():
    """One fit, shared by every assertion below.

    The fit costs a couple of seconds, which is why it is module-scoped, and
    `random_seed` makes it reproducible so that a failure here can be replayed.
    `scipy.optimize.minimize` is wrapped rather than replaced, so the fit is
    real and the recorded calls are the ones gpyreg actually made.
    """
    data = make_data()
    real_minimize = scipy.optimize.minimize
    calls = []

    def recording_minimize(*args, **kwargs):
        calls.append(kwargs)
        return real_minimize(*args, **kwargs)

    scipy.optimize.minimize = recording_minimize
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimiser = Bads(
                model=make_wrapper(data),
                data=data,
                minimisation=minimise.LogLikelihood.continuous,
                ppt_identifier="ppt",
                initial_guess=INITIAL_GUESS,
                display=False,
                options={"display": "off", "max_fun_evals": 100, "random_seed": 42},
            )
            optimiser.optimise()
    finally:
        scipy.optimize.minimize = real_minimize

    return optimiser, calls, caught


class TestBadsSmoke:
    def test_fit_reaches_scipys_bounded_minimiser(self, fitted):
        """The canary: if this stops holding, the fit no longer exercises
        L-BFGS-B and the rest of this file stops guarding against SciPy
        changes to it, even while it keeps passing."""
        _, calls, _ = fitted
        assert calls, "BADS never called scipy.optimize.minimize"
        assert all(call.get("bounds") is not None for call in calls)
        assert all(call.get("jac") for call in calls)

    def test_one_fit_per_participant(self, fitted):
        optimiser, _, _ = fitted
        assert len(optimiser.parameters) == len(PARTICIPANTS)
        assert tuple(fit["ppt"] for fit in optimiser.fit) == PARTICIPANTS

    def test_fits_record_a_finite_loss_and_hessian(self, fitted):
        optimiser, _, _ = fitted
        for fit in optimiser.fit:
            assert np.isfinite(fit["fun"])
            assert np.shape(fit["hessian"]) == (2, 2)
            assert np.all(np.isfinite(fit["hessian"]))

    def test_recovers_the_generating_parameters(self, fitted):
        """Guards against a fit that completes but returns the initial guess:
        BADS must move off `INITIAL_GUESS` and towards the true values."""
        optimiser, _, _ = fitted
        estimates = np.array(
            [[fit["alpha"], fit["beta"]] for fit in optimiser.parameters]
        )
        assert np.allclose(estimates, [TRUE_ALPHA, TRUE_BETA], atol=0.4)
        assert not np.allclose(estimates, INITIAL_GUESS[0], atol=0.05)


class TestBitwiseInvertedOptions:
    """pybads inverts `options["gp_fixed_mean"]` with `~`, which is wrong for a
    Python bool and becomes a `TypeError` on Python 3.16. `Bads` hands the
    option over as a NumPy boolean so the expression stays correct and keeps
    working. See https://github.com/DevComPsy/cpm/issues/88.
    """

    def test_options_are_added_when_the_caller_passes_none(self):
        coerced = numpy_bool_options({})
        assert coerced["options"] == {"gp_fixed_mean": np.bool_(False)}
        assert isinstance(coerced["options"]["gp_fixed_mean"], np.bool_)

    def test_other_options_and_keyword_arguments_survive(self):
        coerced = numpy_bool_options(
            {"options": {"max_fun_evals": 100}, "non_box_cons": None}
        )
        assert coerced["options"]["max_fun_evals"] == 100
        assert coerced["non_box_cons"] is None
        assert isinstance(coerced["options"]["gp_fixed_mean"], np.bool_)

    @pytest.mark.parametrize("value", [True, False], ids=["fixed", "free"])
    def test_a_caller_supplied_value_is_kept_and_coerced(self, value):
        coerced = numpy_bool_options({"options": {"gp_fixed_mean": value}})
        assert coerced["options"]["gp_fixed_mean"] == value
        assert isinstance(coerced["options"]["gp_fixed_mean"], np.bool_)

    def test_the_callers_dict_is_not_mutated(self):
        options = {"max_fun_evals": 100}
        numpy_bool_options({"options": options})
        assert options == {"max_fun_evals": 100}

    def test_a_real_fit_never_inverts_a_python_bool(self, fitted):
        """The load-bearing one. The `DeprecationWarning` marks exactly the
        expression that Python 3.16 turns into a `TypeError`, so zero warnings
        across a full fit means the fit cannot raise there."""
        _, _, caught = fitted
        bitwise = [w for w in caught if "Bitwise inversion" in str(w.message)]
        assert bitwise == []
