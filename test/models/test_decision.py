import pytest
import numpy as np
from cpm.models.decision import Softmax, Sigmoid, GreedyRule, ChoiceKernel


def test_softmax():
    expected = np.array([0.30719589, 0.18632372, 0.50648039])
    activations = np.array([0.1, 0, 0.2])
    softmax = Softmax(temperature=5, activations=activations)
    policies = softmax.compute()
    choice = softmax.choice()
    assert policies.shape == (3,), "The shape of the policies is incorrect."
    assert np.isclose(
        policies.sum(), 1
    ), "The probabilities in the softmax do not sum to 1."
    assert np.allclose(
        policies, expected
    ), "The probabilities in the softmax are incorrect."


def test_softmax_irreducible_noise():
    expected = np.array([0.30980963, 0.20102468, 0.48916569])
    activations = np.array([0.1, 0, 0.2])
    softmax = Softmax(temperature=5, xi=0.1, activations=activations)
    policies = softmax.irreducible_noise()
    assert policies.shape == (3,)
    assert np.isclose(policies.sum(), 1)
    assert np.allclose(
        policies, expected
    ), "The probabilities in the softmax are incorrect."


def test_softmax_choice():
    activations = np.array([0.1, 0, 0.2])
    softmax = Softmax(temperature=1, activations=activations)
    choice = softmax.choice()
    assert choice in [0, 1, 2], "The Softmax.choice output is not in the expected range."


def test_sigmoid():
    expected = np.array([0.52497919, 0.5, 0.549834])
    activations = np.array([0.1, 0, 0.2])
    sigmoid = Sigmoid(temperature=1, activations=activations)
    policies = sigmoid.compute()
    assert policies.shape == (3,)
    assert np.all(policies >= 0) and np.all(policies <= 1)
    assert np.allclose(
        policies, expected
    ), "The probabilities in the softmax are incorrect."


def test_sigmoid_choice():
    activations = np.array([0.1, 0, 0.2])
    sigmoid = Sigmoid(temperature=1, activations=activations)
    choice = sigmoid.choice()
    assert choice in [0, 1, 2], "The Sigmoid.choice output is not in the expected range."


def test_greedy_rule():
    expected = np.array([0.75, 0.25])
    activations = np.array([[0.31, 0, 0.2], [-0.6, 0, 0.9]])
    greedy = GreedyRule(activations=activations, epsilon=0.25)
    policies = greedy.compute()
    assert policies.shape == (2,)
    assert np.isclose(policies.sum(), 1)
    assert np.allclose(
        policies, expected
    ), "The probabilities in the softmax are incorrect."


def test_greedy_rule_choice():
    activations = np.array([[0.1, 0, 0.2], [-0.6, 0, 0.9]])
    greedy = GreedyRule(activations=activations, epsilon=0.1)
    choice = greedy.choice()
    assert choice in [
        0,
        1,
    ], "The GreedyRule.choice output is not in the expected range."


def test_choice_kernel():
    expected = np.array([0.44028635, 0.55971365])
    activations = np.array([[0.1, 0, 0.2], [-0.6, 0, 0.9]])
    kernel = np.array([0.1, 0.9])
    choice_kernel = ChoiceKernel(
        temperature_activations=1,
        temperature_kernel=1,
        activations=activations,
        kernel=kernel,
    )
    policies = choice_kernel.compute()
    assert policies.shape == (2,)
    assert np.isclose(policies.sum(), 1)
    assert np.allclose(
        policies, expected
    ), "The probabilities in the ChoiceKernel are incorrect."


def test_choice_kernel_choice():
    activations = np.array([[0.1, 0, 0.2], [-0.6, 0, 0.9]])
    kernel = np.array([0.1, 0.9])
    choice_kernel = ChoiceKernel(
        temperature_activations=1,
        temperature_kernel=1,
        activations=activations,
        kernel=kernel,
    )
    choice_kernel.compute()
    choice = choice_kernel.choice()
    assert choice in [
        0,
        1,
    ], "The ChoiceKernel.choice output is not in the expected range."


if __name__ == "__main__":
    pytest.main()
