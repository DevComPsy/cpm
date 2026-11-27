"""
testunits for metai.py

tests cover:
- entropy
    entropy is 0 for 0 and 1 and bounded by 0 and 1
    h2(): entropy is smaller for higher and 0 for perfect accuracy
    H2r(): entropy of accuracy is smaller for higher and 0 for perfect accuracy

- conditional entropy:
    conditional entropy (H2r_given_c) is lower for informative confidence 
    (uncertainty about accuracy after seeing confidence lower for informative confidence)
    conditional entropy (H2r_given_c) is bounded by total entropy of accuracy (H2r)

- counts sum to n:
    acc_counts: n_correct and n_incorrect trials sum to total number of trials 
    bin_counts: n_correct and n_incorrect trials sum over all bins sum to total number of trials 

- confidence binning process:
    in case of constant confidence conf_binned flags it as rebin or raises ValueError

- dprime:
    dprime finite for perfect and zero performance, tests if correction works
    dprime higher for better performance
    
"""
import pandas as pd
import numpy as np
import pytest
import warnings

import sys
import os

# adds parent directory to path so that metai.py can be imported directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cpm.utils.metai import h2, acc_counts, H2r, H2r_given_c, conf_binned, bin_counts, dprime

@pytest.fixture
def synthetic_data():
    # Creates a DataFrame with 2 participants with 8 trials each with the required columns
    return pd.DataFrame({
        "participant": [1, 1, 1, 1, 1, 1, 1, 1, 
                        2, 2, 2, 2, 2, 2, 2, 2],
        "signal": [0, 0, 0, 0, 1, 1, 1, 1,
                   0, 0, 0, 0, 1, 1, 1, 1],
        "response": [0, 0, 0, 0, 1, 1, 1, 1,
                     0, 1, 0, 1, 0, 1, 0, 1],
        "confidence": [0.9, 0.8, 0.85, 0.2, 0.9, 0.75, 0.8, 0.15,
                       0.5, 0.5, 0.5,  0.5, 0.5, 0.5,  0.5, 0.5],
        "accuracy": [1, 1, 1, 0, 1, 1, 1, 0,
                     1, 0, 1, 0, 0, 1, 0, 1],
    })

@pytest.fixture
def balanced_data():
    rng = np.random.default_rng(3)
    accuracy = np.array([0] * 50 + [1] * 50)
    confidence = rng.uniform(0, 1, 100)
    return accuracy, confidence

@pytest.fixture
def perfect_acc_data():
    rng = np.random.default_rng(5)
    accuracy = np.ones(100, dtype = int)
    confidence = rng.uniform(0, 1, 100)
    return accuracy, confidence

@pytest.fixture
def informative_conf_data():
    accuracy = np.array([0] * 50 + [1] * 50)
    confidence = np.concatenate([
        np.linspace(0.0, 0.4, 50),
        np.linspace(0.6, 1.0, 50),
    ])
    return accuracy, confidence

#tests entropy and conditional entropy
def test_h2():
    assert h2(0) == pytest.approx(0)
    assert h2(1) == pytest.approx(0)
    assert h2(0.5) == pytest.approx(1.0)

def test_h2_zero_perfect_acc(perfect_acc_data):
    accuracy, _ = perfect_acc_data
    assert h2(np.mean(accuracy)) == 0

def test_higher_acc_lower_entropy(balanced_data):
    accuracy_low, _ = balanced_data
    accuracy_high = np.array([0] * 30 + [1] * 70)
    assert h2(np.mean(accuracy_low)) > h2(np.mean(accuracy_high))

def test_H2r_smaller_lower_accuracy(balanced_data, perfect_acc_data):
    accuracy_low, _ = balanced_data
    n_correct, n_incorrect = acc_counts(accuracy_low)
    entropy_acc_low = H2r(n_correct=n_correct, n_incorrect=n_incorrect)

    accuracy_high, _ = perfect_acc_data
    n_correct, n_incorrect = acc_counts(accuracy_high)
    entropy_acc_high = H2r(n_correct=n_correct, n_incorrect=n_incorrect)
    assert entropy_acc_low > entropy_acc_high

def test_H2r_zero_perfect_acc(perfect_acc_data):
    accuracy, _ = perfect_acc_data
    n_correct, n_incorrect = acc_counts(accuracy)
    assert H2r(n_correct, n_incorrect) == pytest.approx(0.0)

def test_conf_binned_raises_or_rebins_confidence_constant():
    confidence = np.full(100, 0.7)
    try: 
        _, info = conf_binned(confidence, bins = 4)

        assert info["rebin"] == 1, (
                      "identical confidence values shouzld set rebin=1"
                )
    except ValueError:
        pass

def test_cond_entropy_lower_informative_conf(informative_conf_data, balanced_data):
    accuracy, confidence_informative = informative_conf_data
    n_correct_bin, n_incorrect_bin, _ = bin_counts(accuracy, confidence_informative, bins=4)
    cond_entropy_inf_conf = H2r_given_c(n_correct_bin, n_incorrect_bin) 

    _, confidence_uninformative = balanced_data
    n_correct_bin, n_incorrect_bin, _ = bin_counts(accuracy, confidence_uninformative, bins=4)
    cond_entropy_uninf_conf = H2r_given_c(n_correct_bin, n_incorrect_bin) 

    assert cond_entropy_uninf_conf > cond_entropy_inf_conf

def test_cond_entropy_bounded_by_entropy(balanced_data):
    accuracy, confidence = balanced_data
    n_correct, n_incorrect = acc_counts(accuracy)
    full_entropy = H2r(n_correct, n_incorrect)

    n_correct_bin, n_incorrect_bin, _ = bin_counts(accuracy, confidence, bins=4)
    conditional_entropy = H2r_given_c(n_correct_bin, n_incorrect_bin)
    assert full_entropy >= conditional_entropy

#tests count functions
def test_bin_counts_sum_to_n(balanced_data):
    accuracy, confidence = balanced_data
    bin_counts_correct, bin_counts_incorrect, _ = bin_counts(accuracy, confidence, bins=4)
    assert bin_counts_correct.sum() + bin_counts_incorrect.sum() == len(accuracy)

def test_acc_counts_sum_to_n(balanced_data):
    accuracy, _ = balanced_data
    n_correct, n_incorrect = acc_counts(accuracy)
    assert n_correct + n_incorrect == len(accuracy)

#tests dprime
def test_dprime_finite_perfect_performance():
    signal = np.array([1] * 50 + [0] * 50)
    response = np.array([1] * 50 + [0] * 50)
    assert np.isfinite (dprime(signal, response))

def test_dprime_finite_zero_performance():
    signal = np.array([1] * 50 + [0] * 50)
    response = np.array([0] * 50 + [1] * 50)
    assert np.isfinite (dprime(signal, response))

def test_dprime_higher_better_performance(synthetic_data):
    ppt1 = synthetic_data[synthetic_data["participant"] == 1]
    ppt2 = synthetic_data[synthetic_data["participant"] == 2]
    dp1 = dprime(ppt1["signal"].values, ppt1["response"].values)
    dp2 = dprime(ppt2["signal"].values, ppt2["response"].values)
    assert dp1 > dp2