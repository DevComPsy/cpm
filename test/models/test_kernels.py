"""
Every kernel in cpm.models.kernels gives the same result as the class it is
named after, compiled with numba and as plain Python.
"""

import warnings

import numpy as np
import pytest

from cpm.core import _jit
from cpm.models import activation, attention, decision, learning

RNG = np.random.default_rng(2024)
TOLERANCE = dict(rtol=1e-12, atol=1e-13)


@pytest.fixture(params=["numba", "python"], scope="module")
def k(request):
    if request.param == "numba" and not _jit.JIT_ENABLED:
        pytest.skip("numba is not installed or disabled")
    return _jit.kernels("cpm.models.kernels", python=request.param == "python")


def draws(n=25):
    return range(n)


def test_the_python_build_is_not_compiled():
    python = _jit.kernels("cpm.models.kernels", python=True)
    assert not hasattr(python.softmax, "py_func")
    if _jit.JIT_ENABLED:
        assert hasattr(_jit.kernels("cpm.models.kernels").softmax, "py_func")


def test_softmax(k):
    for _ in draws():
        values = RNG.normal(0, 2, RNG.integers(2, 6))
        beta = RNG.uniform(0, 5)
        expected = decision.Softmax(temperature=beta, activations=values).compute()
        np.testing.assert_allclose(k.softmax(values, beta), expected, **TOLERANCE)
        np.testing.assert_allclose(k.log_softmax(values, beta), np.log(expected), **TOLERANCE)
        if len(values) == 2:
            assert k.p_second(beta, values[0], values[1]) == pytest.approx(expected[1], rel=1e-12)


def test_softmax_does_not_overflow(k):
    policy = k.softmax(np.array([1000.0, 999.0]), 2.0)
    assert np.all(np.isfinite(policy))
    assert policy[0] == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert k.p_second(1000.0, 0.0, 1.0) == 1.0
    assert k.p_second(1000.0, 1.0, 0.0) == 0.0


def test_softmax_noise(k):
    for _ in draws():
        values = RNG.normal(0, 2, 3)
        beta, xi = RNG.uniform(0, 5), RNG.uniform(0, 1)
        expected = decision.Softmax(temperature=beta, xi=xi, activations=values).irreducible_noise()
        np.testing.assert_allclose(k.softmax_noise(values, beta, xi), expected, **TOLERANCE)


def test_sigmoid(k):
    for _ in draws():
        activations = RNG.normal(0, 1, 3)
        temperature, bias = RNG.uniform(0, 5), RNG.normal()
        expected = decision.Sigmoid(temperature=temperature, activations=activations, beta=bias).compute()
        np.testing.assert_allclose(k.sigmoid(activations, temperature, bias), expected, **TOLERANCE)


def test_greedy(k):
    for _ in draws():
        activations = RNG.normal(0.5, 1, (4, 3))
        epsilon = RNG.uniform(0, 0.3)
        expected = decision.GreedyRule(activations=activations, epsilon=epsilon).compute()
        np.testing.assert_allclose(k.greedy(activations, epsilon), expected, **TOLERANCE)
    ## all non-positive: uniform
    np.testing.assert_allclose(k.greedy(-np.ones((3, 2)), 0.1), np.full(3, 1 / 3))


def test_choice_kernel(k):
    for _ in draws():
        activations = RNG.normal(0, 1, (3, 2))
        kernel = RNG.uniform(0, 1, 3)
        ta, tk = RNG.uniform(0, 2, 2)
        expected = decision.ChoiceKernel(temperature_activations=ta, temperature_kernel=tk,
                                         activations=activations, kernel=kernel).compute()
        np.testing.assert_allclose(k.choice_kernel(activations, kernel, ta, tk), expected, **TOLERANCE)


def test_choose_reproduces_numpy_random_choice(k):
    for n in (2, 3, 5):
        for _ in draws(200):
            policy = RNG.dirichlet(np.ones(n))
            np.random.seed(int(RNG.integers(1 << 31)))
            state = np.random.get_state()
            expected = np.random.choice(n, p=policy)
            np.random.set_state(state)
            assert k.choose(policy, np.random.random_sample()) == expected
    ## also with the option list and two-element probabilities PTSM2025 draws from
    for _ in draws(200):
        p = RNG.uniform()
        np.random.seed(int(RNG.integers(1 << 31)))
        state = np.random.get_state()
        expected = np.random.choice([0, 1], p=[1 - p, p])
        np.random.set_state(state)
        assert k.choose(np.array([1 - p, p]), np.random.random_sample()) == expected


def test_delta_and_separable_rules(k):
    for _ in draws():
        weights = RNG.normal(0, 1, (2, 4))
        feedback = RNG.normal(0, 1, 2)
        stimulus = RNG.integers(0, 2, 4).astype(float)
        alpha = RNG.uniform()
        rule = learning.DeltaRule(alpha=alpha, weights=weights, feedback=feedback, input=stimulus)
        expected = rule.compute()
        change, error = k.delta_rule(weights, feedback, stimulus, alpha)
        np.testing.assert_allclose(change, expected, **TOLERANCE)
        np.testing.assert_allclose(error, rule.error, **TOLERANCE)

        rule = learning.SeparableRule(alpha=alpha, weights=weights, feedback=feedback, input=stimulus)
        expected = rule.compute()
        change, error = k.separable_rule(weights, feedback, stimulus, alpha)
        np.testing.assert_allclose(change, expected, **TOLERANCE)
        np.testing.assert_allclose(error, rule.error, **TOLERANCE)


def test_q_learning(k):
    for _ in draws():
        values = RNG.normal(0.3, 1, 4)
        reward, maximum = RNG.normal(), RNG.normal()
        alpha, gamma = RNG.uniform(size=2)
        expected = learning.QLearningRule(alpha=alpha, gamma=gamma, values=values,
                                          reward=reward, maximum=maximum).compute()
        np.testing.assert_allclose(k.q_learning(values, reward, maximum, alpha, gamma), expected, **TOLERANCE)


def test_humble_teacher(k):
    for _ in draws():
        weights = RNG.normal(0, 1, (2, 3))
        feedback = RNG.integers(0, 2, 2)
        stimulus = RNG.integers(0, 2, 3).astype(float)
        alpha = RNG.uniform()
        expected = learning.HumbleTeacher(alpha=alpha, weights=weights, feedback=feedback, input=stimulus).compute()
        np.testing.assert_allclose(k.humble_teacher(weights, feedback.astype(float), stimulus, alpha),
                                   expected, **TOLERANCE)


def test_sarsa_trace(k):
    for _ in draws():
        q_mf, q2 = RNG.uniform(0, 5, (2, 2)), RNG.uniform(0, 5, 2)
        s1, a, s2 = RNG.integers(0, 2, 3)
        reward, lr, lam = RNG.uniform(0, 9), RNG.uniform(), RNG.uniform()
        rule = learning.SARSATrace(learning_rate=lr, eligibility_trace=lam, model_free_values=q_mf,
                                   second_stage_values=q2, starting_state=s1, action=a,
                                   reached_second_stage=s2, reward=reward)
        d_mf, d2 = rule.compute()
        new_mf, new_2 = q_mf.copy(), q2.copy()
        pe1, pe2 = k.sarsa_trace_update(new_mf, new_2, s1, a, s2, reward, lr, lam)
        np.testing.assert_allclose(new_mf, q_mf + d_mf, **TOLERANCE)
        np.testing.assert_allclose(new_2, q2 + d2, **TOLERANCE)
        assert pe1 == pytest.approx(rule.stage1_prediction_error, rel=1e-12)
        assert pe2 == pytest.approx(rule.stage2_prediction_error, rel=1e-12)


def test_activations(k):
    for _ in draws():
        stimulus = RNG.integers(0, 2, 3).astype(float)
        stimulus[0] = 1.0
        weights = RNG.normal(0, 1, (2, 3))
        expected = activation.SigmoidActivation(input=stimulus, weights=weights).compute()
        np.testing.assert_allclose(k.sigmoid_activation(stimulus, weights), expected, **TOLERANCE)

        salience, P = RNG.uniform(0.1, 1, 3), RNG.uniform(0.5, 3)
        expected = activation.CompetitiveGating(input=stimulus, values=weights, salience=salience, P=P).compute()
        np.testing.assert_allclose(k.competitive_gating(stimulus, weights, salience, P), expected, **TOLERANCE)

        index, shift = int(RNG.integers(0, 3)), RNG.normal()
        expected = activation.Offset(input=weights[0], offset=shift, index=index).compute()
        np.testing.assert_allclose(k.offset(weights[0], shift, index), expected, **TOLERANCE)


@pytest.mark.parametrize("weighting", ["tk", "power", "prelec", "gw"])
def test_prospect_utility(k, weighting):
    warnings.simplefilter("ignore")
    for _ in draws(50):
        magnitudes = RNG.normal(0, 1, 3)
        magnitudes[0] = 0.0 if RNG.uniform() < 0.2 else magnitudes[0]
        probabilities = RNG.uniform(0, 1, 3)
        probabilities[1] = 1.0
        alpha, beta, lam, gamma, delta = RNG.uniform(0.2, 2, 5)
        expected = activation.ProspectUtility(
            magnitudes=magnitudes, probabilities=probabilities, alpha=alpha, beta=beta,
            lambda_loss=lam, gamma=gamma, delta=delta, weighting=weighting,
        ).compute()
        got = [
            k.prospect_weight(p, m, gamma, delta, k.WEIGHTING[weighting])
            * k.prospect_utility(m, alpha, beta, lam)
            for m, p in zip(magnitudes, probabilities)
        ]
        np.testing.assert_allclose(got, expected, **TOLERANCE)


def test_rapid_attention_shift(k):
    for _ in draws():
        weights = RNG.normal(0, 1, (2, 3))
        predictions, error = RNG.normal(0, 1, 2), RNG.normal(0, 1, 2)
        stimulus, gain = RNG.integers(0, 2, 3).astype(float), RNG.uniform(0.1, 1, 3)
        pnorm, P, rho = RNG.uniform(0.5, 2), RNG.uniform(1, 3), RNG.uniform(0, 1)
        expected = attention.RapidAttentionShift(weights=weights, predictions=predictions, input=stimulus,
                                                 error=error, gain=gain, pnorm=pnorm, P=P, rho=rho).compute()
        np.testing.assert_allclose(
            k.rapid_attention_shift(weights, predictions, stimulus, error, gain, pnorm, P, rho),
            expected, **TOLERANCE,
        )
