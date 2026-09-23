import pytest
import numpy as np
from cpm.models.learning import (
    DeltaRule,
    SeparableRule,
    QLearningRule,
    HumbleTeacher,
    SARSATrace,
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

def test_humble_teacher():
    weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    teacher = np.array([0, 1])
    input = np.array([1, 1, 1])
    humble_teacher = HumbleTeacher(
        alpha=0.1, weights=weights, feedback=teacher, input=input
    )
    humble_teacher.compute()

    computed_weights = humble_teacher.delta
    assert computed_weights.shape == weights.shape
    assert np.allclose(
        computed_weights, np.array([[-0.16, -0.16, -0.16], [0.0, 0.0, 0.0]])
    ), "The weights are not updated correctly with the HumbleTeacher rule."
    assert np.allclose(
        humble_teacher.weights, np.array([[-0.06, 0.04, 0.14], [0.4, 0.5, 0.6]])
    ), "The teacher should be zero after the HumbleTeacher update."


def test_sarsa_trace():
    model_free_values = np.zeros((2, 2))
    second_stage_values = np.array([4.5, 4.5])
    sarsa = SARSATrace(
        learning_rate=0.5,
        eligibility_trace=0.6,
        model_free_values=model_free_values,
        second_stage_values=second_stage_values,
        starting_state=0,
        action=1,
        reached_second_stage=0,
        reward=7.0,
    )
    model_free_delta, planet_value_delta = sarsa.compute()
    assert sarsa.stage1_prediction_error == 4.5
    assert sarsa.stage2_prediction_error == 2.5
    assert np.allclose(
        model_free_delta, np.array([[0.0, 3.0], [0.0, 0.0]])
    ), "The model-free values are not updated correctly with the SARSA rule."
    assert np.allclose(
        planet_value_delta, np.array([1.25, 0.0])
    ), "The second-stage values are not updated correctly with the SARSA rule."
    assert np.all(model_free_values == 0), "The input values should not be modified."


if __name__ == "__main__":
    pytest.main()
