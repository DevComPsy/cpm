import pytest
import pickle
import os
import warnings

from cpm.applications.reinforcement_learning import RLRW
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
            "NaN in policy with parameters: 0.5, 1000, \nand with policy: [ 0. nan]\n"
            in str(w[-1].message)
        ), "Warning message mismatch"

    print("test_model_warnings passed")


if __name__ == "__main__":
    pytest.main()
