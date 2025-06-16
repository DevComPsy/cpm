import numpy as np
import pandas as pd
import pickle
import os
import pytest
from cpm.applications import signal_detection


# Define __file__ manually if it's not already defined
if "__file__" not in globals():
    __file__ = os.path.abspath("test_signal_detection.py")

## load data for regression tests
expected_output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "test_signal_detection.pkl",
)

print("Expected output path:", expected_output_path)

if not os.path.exists(expected_output_path):
    with open(expected_output_path, "wb") as f:
        pickle.dump({}, f)  # Create an empty pickle file if it doesn't exist

with open(expected_output_path, "rb") as f:
    expected_output = pickle.load(f)


@pytest.fixture
def synthetic_data():
    # Create a simple DataFrame with required columns
    return pd.DataFrame({
        "participant": [1, 1, 2, 2],
        "signal": [0, 1, 0, 1],
        "response": [0, 1, 0, 1],
        "confidence": [1, 2, 2, 1],
        "accuracy": [1, 0, 1, 1],
        "observed": [1, 0, 1, 0],
    })

def test_metad_nll_runs():
    guess = np.array([1.0, 0.5, 0.5, 0.5, 0.5])
    nR_S1 = np.array([10, 5, 2, 1, 1, 1])
    nR_S2 = np.array([1, 1, 1, 2, 5, 10])
    nRatings = 3
    d1 = 1.0
    t1c1 = 0.0
    s = 1
    result = signal_detection.metad_nll(
        guess, nR_S1, nR_S2, nRatings, d1, t1c1, s
    )
    assert isinstance(result, float)
    assert np.isclose(result, 1e+300)

def test_fit_metad_runs():
    nR_S1 = np.array([10, 5, 2, 1, 1, 1])
    nR_S2 = np.array([1, 1, 2, 5, 10, 1])
    nRatings = 3
    result = signal_detection.fit_metad(nR_S1, nR_S2, nRatings)
    assert isinstance(result, dict)
    assert "meta_d" in result
    assert "logL" in result

def test_estimatormetad_init_optimise_export(synthetic_data):
    data = synthetic_data
    est = signal_detection.EstimatorMetaD(
        data=data.groupby("participant"),
        bins=2,
        parallel=False,
        display=0,
        ppt_identifier="participant"
    )
    # Should initialize with correct attributes
    assert hasattr(est, "data")
    assert hasattr(est, "bins")
    assert hasattr(est, "parameters")
    # Export should return a DataFrame (even if empty before optimise)
    est.optimise()
    df = est.export()
    assert isinstance(df, pd.DataFrame)
    assert np.all(np.isclose(expected_output, df, atol=1e-5))
