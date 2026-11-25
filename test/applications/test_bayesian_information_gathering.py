"""
`BayesianInformationGathering` reproduces the toolbox of the authors, computes
the equations of Hauser et al. (2017b), and gives the same results with and
without numba.

The toolbox of the authors (https://github.com/DevComPsy/BayesianInformationGathering,
`IST_model_subjectiveC` with the generative belief, `pY`) fitted the first
example participant of `load_information_sampling_data`, with the cost scale
and slope fixed at 10, and saved the parameters below with a negative log
likelihood of 31.40952410330629. The toolbox integrates the probability of the
next card numerically; with its integration, `simpson_transitions`, the model
gives the same log likelihood.
"""

import copy
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.stats import beta, binom

from cpm.applications import information_gathering
from cpm.applications.information_gathering import BayesianInformationGathering
from cpm.core import _jit
from cpm.core.optimisers import objective
from cpm.datasets import load_information_sampling_data
from cpm.optimisation import FminBound, minimise

BACKENDS = ["python", pytest.param("numba", marks=pytest.mark.skipif(
    not _jit.JIT_ENABLED, reason="numba is not installed or disabled"))]
TOLERANCE = dict(rtol=1e-12, atol=1e-12)
CARDS = 25

TOOLBOX_FIT = {
    "cost_scale": 10.0,
    "cost_slope": 10.0,
    "temperature": 1.0000000032886147,
    "lapse_rate": 1.2300819299531378e-03,
    "impatience_fixed": 24.240512110449298,
}
TOOLBOX_NEGATIVE_LOG_LIKELIHOOD = 31.40952410330629

SETTINGS = {
    "cost_scale": [8.0, 0.0, 30.0],
    "cost_slope": [2.0, 0.0, 20.0],
    "impatience_fixed": [14.0, 0.0, 25.0],
    "impatience_decreasing": [5.0, 0.0, 25.0],
    "temperature": [3.0, 1.0, 50.0],
    "lapse_rate": [0.05, 0.0, 0.5],
}


@pytest.fixture(autouse=True)
def quiet():
    warnings.simplefilter("ignore")


def participant(ppt=1):
    data = load_information_sampling_data()
    data = data[data.ppt == ppt].reset_index(drop=True)
    data["observed"] = 1
    return data


def both_conditions():
    """The example games of both participants, those of the second one in the decreasing condition."""
    data = load_information_sampling_data()
    data["condition"] = (data.ppt == 2).astype(int)
    data["observed"] = 1
    return data


def full_games(games=12, seed=0):
    """All 25 cards of each game, in both conditions, as `generate` needs them."""
    rng = np.random.default_rng(seed)
    rows = []
    for game in range(games):
        cards = rng.random(CARDS) < rng.uniform(0.3, 0.7)
        rows.append(pd.DataFrame({
            "game": game,
            "condition": game % 2,
            "n_seen": np.arange(1, CARDS + 1),
            "n_yellow": np.cumsum(cards),
        }))
    return pd.concat(rows, ignore_index=True)


def on_backend(model, backend):
    model._session_model.backend = backend
    return model


def simpson_transitions():
    """`calc_pstate` of the toolbox: the probability of the next card, integrated numerically on 100 points."""
    grid = np.linspace(0, 1, 100)
    middle = (grid[:-1] + grid[1:]) / 2
    out = np.zeros((CARDS + 1, CARDS + 1, 2))
    for seen in range(1, CARDS + 1):
        for yellow in range(seen + 1):
            a, b = 1 + yellow, 1 + seen - yellow
            density = (grid[1:] - grid[:-1]) / 6 * (
                beta.pdf(grid[:-1], a, b) + 4 * beta.pdf(middle, a, b) + beta.pdf(grid[1:], a, b)
            )
            density -= (density.sum() - 1) / len(density)
            out[seen, yellow] = [np.sum(density * (1 - middle)), np.sum(density * middle)]
    return out


def negative_log_likelihood(model):
    model.run()
    return -np.log(model.dependent).sum()


@pytest.mark.parametrize("backend", BACKENDS)
def test_reproduces_the_toolbox_of_the_authors(backend, monkeypatch):
    belief, _ = information_gathering._tables("generative")
    monkeypatch.setattr(information_gathering, "_tables", lambda kind: (belief, simpson_transitions()))
    model = on_backend(BayesianInformationGathering(
        data=participant(1), parameters_settings=TOOLBOX_FIT, belief="generative"), backend)
    assert negative_log_likelihood(model) == pytest.approx(TOOLBOX_NEGATIVE_LOG_LIKELIHOOD, rel=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_exact_transitions_differ_little_from_the_toolbox(backend):
    model = on_backend(BayesianInformationGathering(
        data=participant(1), parameters_settings=TOOLBOX_FIT, belief="generative"), backend)
    assert negative_log_likelihood(model) == pytest.approx(TOOLBOX_NEGATIVE_LOG_LIKELIHOOD, abs=1e-3)


def test_the_beliefs_are_equations_1_2_and_1_3():
    majority, transitions = information_gathering._tables("majority")
    generative, _ = information_gathering._tables("generative")
    for seen in range(1, CARDS + 1):
        for yellow in range(seen + 1):
            a, b = 1 + yellow, 1 + seen - yellow
            if yellow > CARDS / 2:
                expected = 1.0
            elif seen - yellow > CARDS / 2:
                expected = 0.0
            else:
                ## enough yellow cards among the rest, Y - n_y > N_tot / 2 - n_y, given q
                def integrand(q):
                    return binom.sf(CARDS // 2 - yellow, CARDS - seen, q) * beta.pdf(q, a, b)
                expected = integrate.quad(integrand, 0, 1, epsabs=1e-13, epsrel=1e-13)[0]
            assert majority[seen, yellow] == pytest.approx(expected, abs=1e-10)
            expected = integrate.quad(lambda q: beta.pdf(q, a, b), 0.5, 1, epsabs=1e-13)[0]
            assert generative[seen, yellow] == pytest.approx(expected, abs=1e-10)
            ## Equation 1.11: the posterior mean of q
            assert transitions[seen, yellow, 1] == pytest.approx(a / (a + b), rel=1e-15)
            assert transitions[seen, yellow].sum() == pytest.approx(1.0, rel=1e-15)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("belief", ["majority", "generative"])
def test_the_policy_follows_equations_1_8_to_1_13(backend, belief):
    data = both_conditions()
    model = on_backend(BayesianInformationGathering(
        data=data, parameters_settings=SETTINGS, belief=belief), backend)
    model.run()
    out = model.export()
    policy = out[["policy_0", "policy_1", "policy_2"]].to_numpy()
    q = out[["q_values_0", "q_values_1", "q_values_2"]].to_numpy()
    np.testing.assert_allclose(policy.sum(axis=1), 1.0, **TOLERANCE)
    np.testing.assert_array_equal(out["action"], data["action"])
    np.testing.assert_array_equal(out["dependent"], policy[np.arange(len(data)), data["action"]])
    table, _ = information_gathering._tables(belief)
    np.testing.assert_array_equal(out["belief"], table[data.n_seen, data.n_yellow])
    ## Equation 1.8
    np.testing.assert_allclose(q[:, 0], 100 * out["belief"] - 100 * (1 - out["belief"]), **TOLERANCE)
    np.testing.assert_allclose(q[:, 1], -q[:, 0], **TOLERANCE)
    ## Equation 1.13, with the impatience of each condition
    impatience = np.where(data.condition == 0, 14.0, 5.0)
    np.testing.assert_allclose(out["cost"], 8.0 / (1 + np.exp(-2.0 * (data.n_seen - impatience))), **TOLERANCE)
    ## Equation 1.12
    exponentials = np.exp((q - q.max(axis=1, keepdims=True)) / 3.0)
    softmax = exponentials / exponentials.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(policy, softmax * 0.95 + 0.05 / 3, **TOLERANCE)


def backward_induction(belief, cost_scale, cost_slope, impatience, temperature, lapse_rate):
    """Equations 1.8 to 1.13 by backward induction, written out with numpy."""
    table, _ = information_gathering._tables(belief)
    q_open = np.full((CARDS + 1, CARDS + 1), np.nan)
    following = None
    for seen in range(CARDS, 0, -1):
        yellow = np.arange(seen + 1)
        p = table[seen, : seen + 1]
        q_yellow, q_blue = 100 * p - 100 * (1 - p), 100 * (1 - p) - 100 * p
        cost = cost_scale / (1 + np.exp(-cost_slope * (seen - impatience)))
        if following is None:
            q_more = -cost - 10000 + 0 * p
        else:
            p_next = (1 + yellow) / (2 + seen)
            q_more = -cost + (1 - p_next) * following[:-1] + p_next * following[1:]
        q = np.stack([q_yellow, q_blue, q_more], axis=1)
        e = np.exp((q - q.max(axis=1, keepdims=True)) / temperature)
        policy = e / e.sum(axis=1, keepdims=True) * (1 - lapse_rate) + lapse_rate / 3
        q_open[seen, : seen + 1] = q_more
        following = (policy * q).sum(axis=1)
    return q_open


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("belief", ["majority", "generative"])
def test_the_value_of_opening_a_card_is_computed_by_backward_induction(backend, belief):
    data = both_conditions()
    model = on_backend(BayesianInformationGathering(
        data=data, parameters_settings=SETTINGS, belief=belief), backend)
    model.run()
    got = model.export()["q_values_2"].to_numpy()
    for condition, impatience in ((0, 14.0), (1, 5.0)):
        q_open = backward_induction(belief, 8.0, 2.0, impatience, 3.0, 0.05)
        rows = data.condition == condition
        np.testing.assert_allclose(got[rows], q_open[data.n_seen[rows], data.n_yellow[rows]], rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_each_condition_uses_its_own_impatience(backend):
    data = both_conditions()
    model = on_backend(BayesianInformationGathering(data=data, parameters_settings=SETTINGS), backend)
    model.run()
    before = model.export()
    model.reset(parameters={"impatience_decreasing": 20.0})
    model.run()
    after = model.export()
    fixed = (data.condition == 0).to_numpy()
    pd.testing.assert_frame_equal(after[fixed], before[fixed])
    assert not np.allclose(after.loc[~fixed, "policy_2"], before.loc[~fixed, "policy_2"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_per_trial_model_function_reproduces_a_run(backend):
    from cpm.generators import Wrapper

    model = on_backend(BayesianInformationGathering(data=both_conditions(), parameters_settings=SETTINGS), backend)
    per_trial = Wrapper(model=model.model, data=model.data, parameters=model.parameters)
    for wrapper in (model, per_trial):
        wrapper.run()
    pd.testing.assert_frame_equal(per_trial.export(), model.export(), check_exact=False, **TOLERANCE)


def expected_actions(policy, uniforms, n_seen):
    """The actions `generate` takes, from the policy and the uniforms, as `numpy.random.choice` makes them."""
    actions, playing = [], False
    for row, u, seen in zip(policy, uniforms, n_seen):
        playing = playing or seen == 1
        if not playing:
            actions.append(-1)
            continue
        options = row[:2] if seen == CARDS else row
        cdf = np.cumsum(options) / options.sum()
        action = min(int((cdf <= u).sum()), len(options) - 1)
        actions.append(action)
        playing = action == 2
    return np.array(actions)


@pytest.mark.parametrize("backend", BACKENDS)
def test_generate_plays_each_game_until_it_decides(backend):
    data = full_games()
    model = on_backend(BayesianInformationGathering(data=data, parameters_settings=SETTINGS, generate=True), backend)
    np.random.seed(3)
    model.run()
    out = model.export()
    np.random.seed(3)
    uniforms = np.random.random_sample(len(data))
    policy = out[["policy_0", "policy_1", "policy_2"]].to_numpy()
    np.testing.assert_array_equal(out["action"], expected_actions(policy, uniforms, data.n_seen))
    for _, game in out.assign(game=data.game).groupby("game"):
        actions = game["action"].to_numpy()
        decided = np.flatnonzero(actions < 2)[0]
        assert actions[decided] in (0, 1)
        assert (actions[:decided] == 2).all() and (actions[decided + 1:] == -1).all()
        assert game["dependent"].iloc[decided + 1:].isna().all()
    ## the agent does not always decide on the same card
    assert out.groupby(data.game)["action"].apply(lambda a: (a >= 0).sum()).nunique() > 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_generate_decides_after_the_last_card(backend, monkeypatch):
    ## uniforms near 1 choose the last option, opening another card, whenever there is one
    monkeypatch.setattr(information_gathering, "uniforms", lambda n, needed: np.full(n, 0.9999))
    data = full_games(games=2)
    model = on_backend(BayesianInformationGathering(
        data=data, parameters_settings={**SETTINGS, "lapse_rate": 0.3}, generate=True), backend)
    model.run()
    actions = model.export()["action"].to_numpy().reshape(2, CARDS)
    assert (actions[:, :-1] == 2).all()
    assert (actions[:, -1] == 1).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_simulated_actions_can_be_fitted(backend):
    data = full_games()
    simulator = on_backend(BayesianInformationGathering(
        data=data, parameters_settings=SETTINGS, generate=True), backend)
    np.random.seed(11)
    simulator.run()
    simulated = data.assign(action=simulator.export()["action"].to_numpy())
    reached = simulated[simulated.action >= 0].reset_index(drop=True)
    model = on_backend(BayesianInformationGathering(data=reached, parameters_settings=SETTINGS), backend)
    model.run()
    np.testing.assert_allclose(model.dependent[:, 0], simulator.dependent[simulated.action >= 0, 0], **TOLERANCE)


@pytest.mark.skipif(not _jit.JIT_ENABLED, reason="numba is not installed or disabled")
@pytest.mark.parametrize("generate", [False, True])
@pytest.mark.parametrize("belief", ["majority", "generative"])
def test_backends_agree(belief, generate):
    data = full_games() if generate else both_conditions()
    exports = []
    for backend in ("python", "numba"):
        model = on_backend(BayesianInformationGathering(
            data=data, parameters_settings=SETTINGS, belief=belief, generate=generate), backend)
        np.random.seed(7)
        model.run()
        exports.append(model.export())
    pd.testing.assert_frame_equal(exports[0], exports[1], check_exact=False, **TOLERANCE)


def test_a_number_fixes_a_parameter():
    model = BayesianInformationGathering(data=participant(), parameters_settings={"cost_scale": 10, "cost_slope": 10})
    assert set(model.parameters.free()) == {
        "impatience_fixed", "impatience_decreasing", "temperature", "lapse_rate"}
    assert model.parameters.cost_scale == 10.0
    model.run()
    assert np.isfinite(model.dependent).all()


def test_settings_are_checked():
    with pytest.warns(UserWarning, match="default"):
        model = BayesianInformationGathering(data=participant())
    assert model.parameters.temperature.value == 5.0
    with pytest.raises(ValueError, match="belief"):
        BayesianInformationGathering(data=participant(), parameters_settings={}, belief="posterior")
    with pytest.raises(ValueError, match="impatience"):
        BayesianInformationGathering(data=participant(), parameters_settings={"impatience": [1, 0, 2]})


def _changed(data, **columns):
    data = data.copy()
    for key, value in columns.items():
        data.loc[3, key] = value
    return data


def _two_yellow_cards_at_once():
    data = full_games()
    data.loc[0:3, "n_yellow"] = 0
    data.loc[4, "n_yellow"] = 2
    return data


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("data, match", [
    (_changed(participant(), n_seen=0), "n_seen"),
    (_changed(participant(), n_seen=26), "n_seen"),
    (_changed(participant(), n_yellow=5), "n_yellow"),
    (_changed(participant(), condition=2), "condition"),
    (_changed(participant(), action=3), "action"),
    (_changed(participant(), action=np.nan), "action"),
], ids=["n_seen-0", "n_seen-26", "n_yellow", "condition", "action", "action-NaN"])
def test_data_out_of_range_raise_instead_of_reading_past_the_arrays(data, match, backend):
    model = on_backend(BayesianInformationGathering(data=data, parameters_settings=SETTINGS), backend)
    with pytest.raises(IndexError, match=match):
        model.run()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("data, match", [
    (full_games().iloc[:-1], "all 25 cards"),
    (full_games().drop(index=30), "one card per trial"),
    (_two_yellow_cards_at_once(), "one card per trial"),
], ids=["truncated", "skipped", "two-yellow"])
def test_generate_needs_whole_games(data, match, backend):
    model = on_backend(BayesianInformationGathering(
        data=data.reset_index(drop=True), parameters_settings=SETTINGS, generate=True), backend)
    with pytest.raises(ValueError, match=match):
        model.run()


def test_missing_columns_are_reported():
    model = BayesianInformationGathering(data=participant().drop(columns="action"), parameters_settings=SETTINGS)
    with pytest.raises(KeyError, match="action"):
        model.run()
    ## generate does not need the actions
    model = BayesianInformationGathering(data=full_games(), parameters_settings=SETTINGS, generate=True)
    model.run()


def test_models_survive_deepcopy_and_pickling():
    model = BayesianInformationGathering(data=participant(), parameters_settings=SETTINGS)
    model.run()
    clone = pickle.loads(pickle.dumps(copy.deepcopy(model)))
    clone.reset()
    clone.run()
    np.testing.assert_array_equal(clone.dependent, model.dependent)


def test_the_example_participants_can_be_fitted():
    data = load_information_sampling_data()
    data["observed"] = 1
    model = BayesianInformationGathering(
        data=data[data.ppt == 1], parameters_settings={"cost_scale": 10, "cost_slope": 10})
    fit = FminBound(
        model=model,
        data=data,
        minimisation=minimise.LogLikelihood.bernoulli,
        ppt_identifier="ppt",
        initial_guess=[[20.0, 12.5, 2.0, 0.01]],
        number_of_starts=1,
        approx_grad=True,
    )
    fit.optimise()
    result = fit.export().sort_values("ppt")
    assert np.isfinite(result["fun"]).all()
    ## the fit is no worse than the starting point
    for ppt, fun in zip(result["ppt"], result["fun"]):
        one = data[data.ppt == ppt]
        model.reset(data=one)
        start = objective(np.array([20.0, 12.5, 2.0, 0.01]), model, one.observed.to_numpy(),
                          minimise.LogLikelihood.bernoulli, False)
        assert fun <= start + 1e-9
