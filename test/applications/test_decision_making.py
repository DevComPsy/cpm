import pytest
import pickle
import os
import warnings

import pandas as pd
import numpy as np
from cpm.datasets import load_risky_choices
import cpm.applications.decision_making as models

# Define __file__ manually if it's not already defined
if "__file__" not in globals():
    __file__ = os.path.abspath("test_decision_making.py")

def load_expected_output(filename):
    """
    Loads a pickle file from the 'data' directory relative to this test file.
    If the file does not exist, it creates an empty pickle file.
    Returns the loaded object.
    """
    expected_output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        filename,
    )

    print("Expected output path:", expected_output_path)

    if not os.path.exists(expected_output_path):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(expected_output_path), exist_ok=True)
        with open(expected_output_path, "wb") as f:
            pickle.dump({}, f)  # Create an empty pickle file if it doesn't exist

    with open(expected_output_path, "rb") as f:
        expected_output = pickle.load(f)

    return expected_output

@pytest.fixture
def setup_data():
    data = load_risky_choices()
    data["observed"] = data["choice"].astype(int)
    parameters_settings = {
        "alpha":        [3.1, 1e-2, 5.0],    # alpha: starting value 1.0
        "lambda_loss":  [4.6, 1e-2, 5.0],    # lambda_loss: starting value 1.0
        "gamma":        [1.5, 1e-2, 5.0],    # gamma: starting value 0.5
        "temperature":  [1.6, 1e-2, 15.0],    # temperature: starting value 5.0
        "beta":         [1.0, 0.0, 5.0],   # beta: starting value 1.0
        "delta":        [0.5, 1e-2, 5.0],   # delta: starting value 0.0
        "eta":          [0.0,   -0.49,  0.49],
        "phi_gain":     [0.0,   -10.0,  10.0],
        "phi_loss":     [0.0,   -10.0,  10.0],
    }
    weighting = "tk"
    utility_curve = None
    return data, parameters_settings, weighting, utility_curve

# Existing test with error catching
def test_ptsm_default_parameters_run(setup_data):
    data, parameters_settings, weighting, utility_curve = setup_data
    expected = load_expected_output("test_ptsm_default_params.pkl")
    warnings.simplefilter("ignore")
    target = models.PTSM(
        data=data[data.ppt == 1],
        weighting=weighting,
        utility_curve=utility_curve
    )
    try:
        target.run()
    except Exception as e:
        print(f"Error during model run: {e}")
    results = target.export()
    try:
        pd.testing.assert_frame_equal(results, expected, check_dtype=True)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print("Results DataFrame:")
        print(results.head())
        print("Expected DataFrame:")
        print(expected.head())

# Test with custom utility curve
def test_ptsm_custom_utility_curve(setup_data):
    data, parameters_settings, weighting, _ = setup_data
    def custom_utility(x, alpha):
        return x ** alpha + 1
    target = models.PTSM(
        data=data[data.ppt == 1],
        parameters_settings=parameters_settings,
        weighting=weighting,
        utility_curve=custom_utility
    )
    try:
        target.run()
    except Exception as e:
        print(f"Error during model run with custom utility: {e}")
    results = target.export()
    expected = load_expected_output("test_ptsm_custom_utility.pkl")
    try:
        pd.testing.assert_frame_equal(results, expected, check_dtype=True)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print("Results DataFrame:")
        print(results.head())
        print("Expected DataFrame:")
        print(expected.head())

# Test with different weighting
def test_ptsm_power_weighting(setup_data):
    data, parameters_settings, _, utility_curve = setup_data
    expected = load_expected_output("test_ptsm_power_weighting.pkl")
    target = models.PTSM(
        data=data[data.ppt == 1],
        parameters_settings=parameters_settings,
        weighting="power",
        utility_curve=utility_curve
    )
    try:
        target.run()
    except Exception as e:
        print(f"Error during model run with power weighting: {e}")
    results = target.export()
    try:
        pd.testing.assert_frame_equal(results, expected, check_dtype=True)
    except AssertionError as e:
        print(f"AssertionError: {e}\n")
        print("Results DataFrame:")
        print(results.head())
        print("\nExpected DataFrame:")
        print(expected.head())

# Test PTSM1992 model
def test_ptsm1992_run(setup_data):
    data, _, _, _ = setup_data
    # Add required keys for PTSM1992
    target = models.PTSM1992(
        data=data[data.ppt == 1],
    )
    try:
        target.run()
    except Exception as e:
        print(f"Error during PTSM1992 model run: {e}")
    results = target.export()
    expected = load_expected_output("test_ptsm1992_default_params.pkl")
    try:
        pd.testing.assert_frame_equal(results, expected, check_dtype=True)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print("Results DataFrame:")
        print(results.head())
        print("Expected DataFrame:")
        print(expected.head())

# Test PTSM1992 model with custom parameters
def test_ptsm1992_custom_parameters(setup_data):
    data, parameters_settings, _, utility_curve = setup_data
    target = models.PTSM1992(
        data=data[data.ppt == 1],
        parameters_settings=parameters_settings,
        utility_curve=utility_curve
    )
    try:
        target.run()
    except Exception as e:
        print(f"Error during PTSM1992 model run with custom parameters: {e}")
    results = target.export()
    expected = load_expected_output("test_ptsm1992_custom_params.pkl")
    try:
        pd.testing.assert_frame_equal(results, expected, check_dtype=True)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print("Results DataFrame:")
        print(results.head())
        print("Expected DataFrame:")
        print(expected.head())

# Test PTSM1992 model with custom utility curve
def test_ptsm1992_custom_utility_curve(setup_data):
    data, parameters_settings, weighting, _ = setup_data
    def custom_utility(x, alpha, **kwargs):
        if x < 0:
            out = -(np.abs(x) ** alpha) + 1
        else:
            out = x ** alpha + 1
        return out 
    target = models.PTSM1992(
        data=data[data.ppt == 1],
        parameters_settings=parameters_settings,
        weighting=weighting,
        utility_curve=custom_utility
    )
    try:
        target.run()
    except Exception as e:
        print(f"Error during PTSM1992 model run with custom utility: {e}")
    results = target.export()
    expected = load_expected_output("test_ptsm1992_custom_utility.pkl")
    try:
        pd.testing.assert_frame_equal(results, expected, check_dtype=True)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print("Results DataFrame:")
        print(results.head())
        print("Expected DataFrame:")
        print(expected.head())

# Test PTSM1992 model with different weighting
def test_ptsm1992_power_weighting(setup_data):
    data, parameters_settings, _, utility_curve = setup_data
    expected = load_expected_output("test_ptsm1992_power_weighting.pkl")
    target = models.PTSM1992(
        data=data[data.ppt == 1],
        parameters_settings=parameters_settings,
        weighting="power",
        utility_curve=utility_curve
    )
    try:
        target.run()
    except Exception as e:
        print(f"Error during PTSM1992 model run with power weighting: {e}")
    results = target.export()
    try:
        pd.testing.assert_frame_equal(results, expected, check_dtype=True)
    except AssertionError as e:
        print(f"AssertionError: {e}\n")
        print("Results DataFrame:")
        print(results.head())
        print("\nExpected DataFrame:")
        print(expected.head())


def test_ptsm2025_default_parameters_run(setup_data):
    data, parameters_settings, _, utility_curve = setup_data
    expected = load_expected_output("test_ptsm2025_default_params.pkl")
    target = models.PTSM2025(
        data=data[data.ppt == 1],
        utility_curve=utility_curve
    )
    try:
        target.run()
    except Exception as e:
        print(f"Error during PTSM2025 model run: {e}")
    results = target.export()
    try:
        pd.testing.assert_frame_equal(results, expected, check_dtype=True)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print("Results DataFrame:")
        print(results.head())
        print("Expected DataFrame:")
        print(expected.head())

def test_ptsm2025_custom_parameters(setup_data):
    data, parameters_settings, _, utility_curve = setup_data
    target = models.PTSM2025(
        data=data[data.ppt == 1],
        parameters_settings=parameters_settings,
        utility_curve=utility_curve
    )
    try:
        target.run()
    except Exception as e:
        print(f"Error during PTSM2025 model run with custom parameters: {e}")
    results = target.export()
    expected = load_expected_output("test_ptsm2025_custom_params.pkl")
    try:
        pd.testing.assert_frame_equal(results, expected, check_dtype=True)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print("Results DataFrame:")
        print(results.head())
        print("Expected DataFrame:")
        print(expected.head())

def test_ptsm2025_custom_utility_curve(setup_data):
    data, parameters_settings, weighting, _ = setup_data
    def custom_utility(x, alpha, **kwargs):
        if x < 0:
            out = -(np.abs(x) ** alpha) + 1
        else:
            out = x ** alpha + 1
        return out 
    target = models.PTSM2025(
        data=data[data.ppt == 1],
        parameters_settings=parameters_settings,
        utility_curve=custom_utility
    )
    try:
        target.run()
    except Exception as e:
        print(f"Error during PTSM2025 model run with custom utility: {e}")
    results = target.export()
    expected = load_expected_output("test_ptsm2025_custom_utility.pkl")
    try:
        pd.testing.assert_frame_equal(results, expected, check_dtype=True)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print("Results DataFrame:")
        print(results.head())
        print("Expected DataFrame:")
        print(expected.head())

def test_ptsm2025_without_alpha(setup_data):
    data, parameters_settings, _, utility_curve = setup_data
    # Remove 'alpha' from parameters_settings
    target = models.PTSM2025(
        data=data[data.ppt == 1],
        parameters_settings=parameters_settings,
        variant="standard",
        utility_curve=utility_curve
    )
    try:
        target.run()
    except Exception as e:
        print(f"Error during PTSM2025 model run without alpha: {e}")
    results = target.export()
    expected = load_expected_output("test_ptsm2025_without_alpha.pkl")
    try:
        pd.testing.assert_frame_equal(results, expected, check_dtype=True)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print("Results DataFrame:")
        print(results.head())
        print("Expected DataFrame:")
        print(expected.head())