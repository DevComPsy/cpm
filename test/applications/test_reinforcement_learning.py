import pytest
import pickle
import os
import warnings

import numpy as np
import pandas as pd

from cpm.applications.reinforcement_learning import RLRW, HybridMBMF
from cpm.datasets import load_bandit_data

# Define __file__ manually if it's not already defined
if "__file__" not in globals():
    __file__ = os.path.abspath("test_reinforcement_learning.py")

## load data for regression tests
expected_output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "test_rlrw_expected_output.pkl",
)

print("Expected output path:", expected_output_path)

if not os.path.exists(expected_output_path):
    with open(expected_output_path, "wb") as f:
        pickle.dump({}, f)  # Create an empty pickle file if it doesn't exist

with open(expected_output_path, "rb") as f:
    expected_output = pickle.load(f)


@pytest.fixture
def setup_data():
    data = load_bandit_data()
    dimensions = 4
    parameters_settings = [[0.5, 0, 1], [5, 1, 10]]
    return data, dimensions, parameters_settings


def test_default_parameters(setup_data):
    data, dimensions, _ = setup_data
    warnings.simplefilter("ignore")
    model = RLRW(data=data[data.ppt == 1], dimensions=dimensions)
    assert model is not None, "Model initialization with default parameters failed"
    assert model.parameters.alpha.value == 0.5, "Default alpha parameter value mismatch"
    assert (
        model.parameters.temperature.value == 5
    ), "Default temperature parameter value mismatch"
    print("test_default_parameters passed")


def test_custom_parameters(setup_data):
    data, dimensions, parameters_settings = setup_data
    model = RLRW(
        data=data[data.ppt == 1],
        dimensions=dimensions,
        parameters_settings=parameters_settings,
    )
    assert model is not None, "Model initialization with custom parameters failed"
    assert model.parameters.alpha.value == 0.5, "Custom alpha parameter value mismatch"
    assert (
        model.parameters.temperature.value == 5
    ), "Custom temperature parameter value mismatch"
    print("test_custom_parameters passed")


def test_run_model(setup_data):
    data, dimensions, _ = setup_data
    warnings.simplefilter("ignore")
    model = RLRW(data=data[data.ppt == 1], dimensions=dimensions)
    model.run()
    result = model.simulation[0]
    assert result is not None, "Model run failed"
    assert "policy" in result, "Policy not in result"
    assert "reward" in result, "Reward not in result"
    assert "values" in result, "Values not in result"
    assert "change" in result, "Change not in result"
    assert "dependent" in result, "Dependent not in result"
    print("test_run_model passed")


def test_model_accuracy(setup_data):
    data, dimensions, _ = setup_data
    ## suppress warnings
    warnings.simplefilter("ignore")
    model = RLRW(data=data[data.ppt == 1], dimensions=dimensions)
    model.run()
    import numpy as np

    assert np.array_equal(
        model.dependent, expected_output
    ), "Model accuracy test failed"
    print("test_model_accuracy passed")


def test_model_warnings(setup_data):
    data, dimensions, _ = setup_data
    data["observed"] = data["response"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = RLRW(
            data=data[data.ppt == 1],
            dimensions=dimensions,
        )
        assert len(w) > 0, "No warning raised"
        assert issubclass(w[-1].category, UserWarning), "Warning is not a UserWarning"
        assert "No parameters specified, using default parameters." in str(
            w[-1].message
        ), "Warning message mismatch"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = RLRW(
            data=data[data.ppt == 1],
            dimensions=dimensions,
            parameters_settings=[[0.5, 0, 1], [1000, 1, 10]],
        )
        model.run()
        assert len(w) > 0, "Warnings not raised"
        assert issubclass(w[-1].category, UserWarning), "Warning is not a UserWarning"
        assert (
            "NaN values found in policies. Replacing NaN values with 1 and normalising the policies to sum to 1."
            in str(w[-1].message)
        ), "Warning message mismatch"

    print("test_model_warnings passed")


@pytest.fixture
def two_step_data():
    rng = np.random.default_rng(0)
    n = 40
    action = rng.integers(0, 2, n)
    data = pd.DataFrame(
        {
            "s1": rng.integers(0, 2, n),
            "stimuli_first": rng.integers(0, 2, n),
            "action": action,
            "s2": 1 - action,
            "reward": rng.integers(0, 10, n) / 9,
            "reward_0": rng.integers(0, 10, n) / 9,
            "reward_1": rng.integers(0, 10, n) / 9,
            "observed": action,
        }
    )
    settings = [[2, 0, 5], [0.5, 0, 1], [0.5, 0, 1], [0.5, 0, 1], [0.5, -5, 5], [0.5, -5, 5]]
    return data, settings


def test_hybrid_mbmf_default_parameters(two_step_data):
    data, _ = two_step_data
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = HybridMBMF(data=data)
        assert "No parameters specified, using default parameters." in str(
            w[-1].message
        ), "Warning message mismatch"
    assert model.parameters.free() == [
        "inv_temperature",
        "learning_rate",
        "eligibility_trace",
        "mb_weight",
        "choice_stickiness",
        "response_stickiness",
    ], "Free parameters mismatch"
    assert model.parameters.inv_temperature.value == 2
    assert model.parameters.mb_weight.value == 0.5


def test_hybrid_mbmf_run(two_step_data):
    data, settings = two_step_data
    model = HybridMBMF(data=data, parameters_settings=settings)
    model.run()
    assert model.dependent.shape == (len(data), 1)
    assert np.all((model.dependent > 0) & (model.dependent < 1))
    result = model.simulation[0]
    for key in ["policy", "q_mf", "q2", "m", "r", "dependent"]:
        assert key in result, f"{key} not in result"
    assert np.isclose(result["policy"].sum(), 1)
    ## first trial: all values are equal and there is no stickiness yet
    assert np.allclose(result["policy"], 0.5)
    ## observed actions are used when not generating
    actions = np.array([trial["action"] for trial in model.simulation])
    assert np.array_equal(actions, data.action.to_numpy())


def test_hybrid_mbmf_value_updates(two_step_data):
    data, settings = two_step_data
    model = HybridMBMF(data=data.iloc[:1], parameters_settings=settings, q_init=0.5)
    model.run()
    trial = data.iloc[0]
    q2 = model.simulation[0]["q2"]
    expected = 0.5 + 0.5 * (trial.reward - 0.5)
    s2 = int(trial.s2)
    assert np.isclose(q2[s2], expected), "Second-stage value update mismatch"
    assert np.isclose(q2[1 - s2], 0.5), "Unvisited state should not change"


def test_hybrid_mbmf_generate(two_step_data):
    data, settings = two_step_data
    np.random.seed(0)
    model = HybridMBMF(
        data=data[["s1", "stimuli_first", "reward_0", "reward_1"]],
        parameters_settings=settings,
        generate=True,
    )
    model.run()
    for trial, (_, row) in zip(model.simulation, data.iterrows()):
        assert trial["s2"] == 1 - trial["action"]
        assert trial["reward"] == row[f"reward_{trial['s2']}"]
        assert trial["position"] == int(row.stimuli_first) ^ trial["action"]


def test_hybrid_mbmf_pure_model_based_ignores_model_free(two_step_data):
    data, _ = two_step_data
    settings = [[3, 0, 5], [0.5, 0, 1], [0.5, 0, 1], [1, 0, 1], [0, -5, 5], [0, -5, 5]]
    model = HybridMBMF(data=data, parameters_settings=settings)
    model.run()
    ## with w = 1 and no stickiness, P(action 1) depends only on the second-stage values
    q2_before = [np.full(2, 0.5)] + [trial["q2"] for trial in model.simulation[:-1]]
    for q2, trial in zip(q2_before, model.simulation):
        ## action 1 leads to state 0, action 0 leads to state 1
        expected = 1 / (1 + np.exp(-3 * (q2[0] - q2[1])))
        assert np.isclose(trial["dependent"][0], expected)


if __name__ == "__main__":
    pytest.main()
