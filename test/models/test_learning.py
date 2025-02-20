import pytest
import numpy as np
from cpm.models.learning import (
    DeltaRule,
    SeparableRule,
    QLearningRule,
    KernelUpdate,
    HumbleTeacher,
)


def test_delta_rule():
    weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    teacher = np.array([1, 0])
    input = np.array([1, 1, 0])
    delta_rule = DeltaRule(
        alpha=0.1, zeta=0.1, weights=weights, feedback=teacher, input=input
    )
    computed_weights = delta_rule.compute()
    assert computed_weights.shape == weights.shape
    assert np.allclose(
        computed_weights, np.array([[0.07, 0.07, 0.0], [-0.09, -0.09, 0.0]])
    ), "The weights are not updated correctly with the summed delta rule."


def test_delta_rule_noisy_learning_rule():
    weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    teacher = np.array([1, 0])
    input = np.array([1, 1, 0])
    delta_rule = DeltaRule(
        alpha=0.1, zeta=0.1, weights=weights, feedback=teacher, input=input
    )
    computed_weights = delta_rule.noisy_learning_rule()
    assert computed_weights.shape == weights.shape
    assert not np.allclose(
        computed_weights, np.array([[0.07, 0.07, 0.0], [-0.09, -0.09, 0.0]])
    ), "The weights are not updated correctly with the noisy learning rule."


def test_separable_rule():
    weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    teacher = np.array([1, 0])
    input = np.array([1, 1, 0])
    separable_rule = SeparableRule(
        alpha=0.1, zeta=0.1, weights=weights, feedback=teacher, input=input
    )
    computed_weights = separable_rule.compute()
    assert computed_weights.shape == weights.shape
    assert np.allclose(
        computed_weights, np.array([[0.09, 0.08, 0.0], [-0.04, -0.05, 0.0]])
    ), "The weights are not updated correctly with the separable delta rule."


def test_q_learning_rule():
    values = np.array([1, 0.5, 0.99])
    q_learning_rule = QLearningRule(
        alpha=0.1, gamma=0.8, values=values, reward=1, maximum=10
    )
    computed_values = q_learning_rule.compute()
    assert (
        computed_values.shape == values.shape
    ), "The shape of the Q-values is incorrect."
    assert np.allclose(
        computed_values, np.array([1.8, 1.35, 1.791])
    ), "The Q-values are not updated correctly."


def test_kernel_update():
    response = np.array([0, 1, 0, 0])
    alpha = 0.1
    kernel = np.array([0.2, 0.3, 0.4, 0.5])
    input = np.array([1, 1, 0, 0])
    kernel_update = KernelUpdate(
        response=response, alpha=alpha, kernel=kernel, input=input
    )
    computed_kernel = kernel_update.compute()
    assert computed_kernel.shape == kernel.shape
    assert np.allclose(
        computed_kernel, np.array([-0.02, 0.07, 0.0, 0.0])
    ), "The kernel is not updated correctly with the KernelUpdate rule."


def test_humble_teacher():
    weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    teacher = np.array([0, 1])
    input = np.array([1, 1, 1])
    humble_teacher = HumbleTeacher(
        alpha=0.1, weights=weights, feedback=teacher, input=input
    )
    computed_weights = humble_teacher.compute()
    assert computed_weights.shape == weights.shape
    assert np.allclose(
        computed_weights, np.array([[-0.06, -0.06, -0.06], [0.0, 0.0, 0.0]])
    ), "The weights are not updated correctly with the HumbleTeacher rule."


if __name__ == "__main__":
    pytest.main()
