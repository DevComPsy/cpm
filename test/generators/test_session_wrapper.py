"""
A SessionWrapper has to be a drop-in replacement for a Wrapper: the same model
written per trial and per session must give the same dependent variable,
export, log likelihood and fits, and work with Simulator and cpm.hierarchical.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from cpm.core.optimisers import objective
from cpm.generators import Parameters, SessionWrapper, Simulator, Value, Wrapper
from cpm.hierarchical import EmpiricalBayes
from cpm.optimisation import FminBound, Minimize, minimise


def parameters():
    return Parameters(
        alpha=Value(value=0.3, lower=0, upper=1, prior="truncated_normal",
                    args={"mean": 0.5, "sd": 0.25}),
        beta=Value(value=2.0, lower=0, upper=10, prior="truncated_normal",
                   args={"mean": 5, "sd": 2.5}),
        values=np.array([0.5, 0.5]),
    )


def trial_model(parameters, trial):
    """Q-learning on a two-armed bandit with a softmax policy, one trial."""
    values = np.asarray(parameters.values).copy()
    beta = parameters.beta.value
    policy = np.exp(beta * values) / np.sum(np.exp(beta * values))
    choice = int(trial["choice"])
    error = trial["reward"] - values[choice]
    values[choice] += parameters.alpha.value * error
    return {
        "policy": policy,
        "error": error,
        "values": values.copy(),
        "dependent": np.array([policy[1]]),
    }


def session_model(parameters, data):
    """The same model, all trials at once."""
    alpha, beta = parameters.alpha.value, parameters.beta.value
    values = np.asarray(parameters.values, dtype=float).copy()
    n = len(data["choice"])
    policy = np.empty((n, 2))
    error = np.empty(n)
    history = np.empty((n, 2))
    for t in range(n):
        policy[t] = np.exp(beta * values) / np.sum(np.exp(beta * values))
        choice = int(data["choice"][t])
        error[t] = data["reward"][t] - values[choice]
        values[choice] += alpha * error[t]
        history[t] = values
    return {"policy": policy, "error": error, "values": history, "dependent": policy[:, 1]}


def make_data(participants=(1, 2, 3), trials=60, seed=3):
    rng = np.random.default_rng(seed)
    frames = []
    for ppt in participants:
        choice = rng.integers(0, 2, trials)
        frames.append(pd.DataFrame({
            "ppt": ppt,
            "choice": choice,
            "reward": (rng.random(trials) < np.where(choice == 1, 0.7, 0.3)).astype(float),
            "observed": choice,
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def data():
    return make_data()


@pytest.fixture
def one(data):
    return data[data.ppt == 1].reset_index(drop=True)


def both(one):
    warnings.simplefilter("ignore")
    return (
        Wrapper(model=trial_model, data=one, parameters=parameters()),
        SessionWrapper(model=session_model, data=one, parameters=parameters()),
    )


def test_run_gives_the_same_dependent_and_final_state(one):
    trial, session = both(one)
    trial.run()
    session.run()
    assert session.dependent.shape == trial.dependent.shape == (len(one), 1)
    np.testing.assert_allclose(session.dependent, trial.dependent, rtol=0, atol=1e-12)
    np.testing.assert_allclose(session.parameters.values.value, trial.parameters.values.value,
                               rtol=0, atol=1e-12)


def test_export_has_the_same_layout(one):
    trial, session = both(one)
    trial.run()
    session.run()
    pd.testing.assert_frame_equal(session.export(), trial.export(), check_exact=False,
                                  rtol=0, atol=1e-12)


def test_objective_is_the_same_before_and_after_reset(one):
    trial, session = both(one)
    observed = one.observed.to_numpy()
    for x in ([0.3, 2.0], [0.8, 6.0], [0.1, 0.5], [0.3, 2.0]):
        expected = objective(x, trial, observed, minimise.LogLikelihood.bernoulli, True)
        got = objective(x, session, observed, minimise.LogLikelihood.bernoulli, True)
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)
    ## the initial state is restored on reset
    np.testing.assert_array_equal(session.__init_parameters__.values.value, [0.5, 0.5])


def test_reset_with_new_data_prepares_them_once(one, data):
    calls = []

    def prepare(frame):
        calls.append(len(frame))
        return {k: frame[k].to_numpy() for k in frame.columns}

    warnings.simplefilter("ignore")
    session = SessionWrapper(model=session_model, data=one, parameters=parameters(), prepare=prepare)
    session.run()
    session.run()
    two = data[data.ppt == 2].reset_index(drop=True).iloc[:30]
    session.reset(data=two)
    session.run()
    assert calls == [len(one), 30]
    assert session.dependent.shape == (30, 1)
    assert len(session.export()) == 30


def test_dict_data_and_scalar_outputs():
    warnings.simplefilter("ignore")

    def model(parameters, data):
        return {"constant": 1.5, "dependent": data["x"] * parameters.a.value}

    session = SessionWrapper(
        model=model,
        data={"ppt": 7, "x": np.arange(4.0), "observed": np.zeros(4)},
        parameters=Parameters(a=Value(value=2.0, lower=0, upper=5, prior="uniform")),
    )
    session.run()
    table = session.export()
    assert list(table.columns) == ["constant", "dependent", "ppt"]
    np.testing.assert_array_equal(table.constant, 1.5)
    np.testing.assert_array_equal(table.dependent, [0, 2, 4, 6])


def test_wrong_number_of_rows_is_an_error(one):
    warnings.simplefilter("ignore")
    session = SessionWrapper(
        model=lambda parameters, data: {"dependent": np.zeros(3)},
        data=one,
        parameters=parameters(),
    )
    with pytest.raises(ValueError, match="rows"):
        session.run()


@pytest.mark.parametrize("optimiser", [FminBound, Minimize])
def test_fits_reach_the_same_optima(data, optimiser):
    warnings.simplefilter("ignore")
    fits = []
    for wrapper in both(data[data.ppt == 1].reset_index(drop=True)):
        fit = optimiser(
            model=wrapper,
            data=data,
            minimisation=minimise.LogLikelihood.bernoulli,
            prior=True,
            ppt_identifier="ppt",
            initial_guess=[[0.4, 3.0]],
            number_of_starts=1,
            **({"approx_grad": True} if optimiser is FminBound else {}),
        )
        fit.optimise()
        fits.append(fit.export())
    trial, session = fits
    for column in ("x_0", "x_1", "fun", "log_likelihood", "log_prior"):
        np.testing.assert_allclose(session[column], trial[column], rtol=1e-6, atol=1e-8)


def test_bads_fit_reaches_the_same_optimum(data):
    bads = pytest.importorskip("cpm.optimisation").Bads
    warnings.simplefilter("ignore")
    fits = []
    for wrapper in both(data[data.ppt == 1].reset_index(drop=True)):
        fit = bads(
            model=wrapper,
            data=data[data.ppt == 1],
            minimisation=minimise.LogLikelihood.bernoulli,
            prior=True,
            ppt_identifier="ppt",
            initial_guess=[[0.4, 3.0]],
            display=False,
            options={"display": "off", "max_fun_evals": 60, "random_seed": 1},
        )
        fit.optimise()
        fits.append(fit.export())
    trial, session = fits
    np.testing.assert_allclose(session["fun"], trial["fun"], rtol=1e-6)


def test_simulator_exports_have_the_same_shape(data):
    warnings.simplefilter("ignore")
    grouped = data.groupby("ppt")
    draws = pd.DataFrame({"alpha": [0.2, 0.5, 0.8], "beta": [1.0, 3.0, 6.0]})
    out = []
    for wrapper in both(data[data.ppt == 1].reset_index(drop=True)):
        simulator = Simulator(wrapper=wrapper, data=grouped, parameters=draws)
        simulator.run()
        out.append(simulator.export())
    trial, session = out
    assert list(session.columns) == list(trial.columns)
    assert session.shape == trial.shape
    pd.testing.assert_frame_equal(session, trial, check_exact=False, rtol=0, atol=1e-12)


def test_empirical_bayes_runs_on_a_session_wrapper(data):
    warnings.simplefilter("ignore")
    results = []
    for wrapper in both(data[data.ppt == 1].reset_index(drop=True)):
        fit = FminBound(
            model=wrapper,
            data=data,
            minimisation=minimise.LogLikelihood.bernoulli,
            prior=True,
            ppt_identifier="ppt",
            initial_guess=[[0.4, 3.0]],
            number_of_starts=1,
            maxiter=30,
            approx_grad=True,
        )
        eb = EmpiricalBayes(optimiser=fit, iteration=2, tolerance=1e-6, chain=1, quiet=True)
        np.random.seed(11)  # EmpiricalBayes draws new starting points on every iteration
        eb.optimise()
        results.append(eb.hyperparameters)
    trial, session = results
    np.testing.assert_allclose(
        session.select_dtypes("number").to_numpy(),
        trial.select_dtypes("number").to_numpy(),
        rtol=1e-6, atol=1e-8,
    )
