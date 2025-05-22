import numpy as np
import pandas as pd
import pytest
from cpm.applications import signal_detection

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

def test_fit_metad_runs():
    nR_S1 = np.array([10, 5, 2, 1, 1, 1])
    nR_S2 = np.array([1, 1, 2, 5, 10, 1])
    nRatings = 3
    result = signal_detection.fit_metad(nR_S1, nR_S2, nRatings)
    assert isinstance(result, dict)
    assert "meta_d" in result
    assert "logL" in result

def test_estimatormetad_init_and_export(synthetic_data):
    est = signal_detection.EstimatorMetaD(
        data=synthetic_data,
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
    df = est.export()
    assert isinstance(df, pd.DataFrame)

def test_estimatormetad_optimise_and_export(synthetic_data):
    est = signal_detection.EstimatorMetaD(
        data=synthetic_data,
        bins=2,
        parallel=False,
        display=0,
        ppt_identifier="participant"
    )
    est.optimise()
    df = est.export()
    assert isinstance(df, pd.DataFrame)