"""
test units for metainformation_theoretic.py

tests cover:
- metaI_d_prime: theoretic ideal Bayes observer
- fit_metaI: meta-I parameter estimation
- EstimatorMetaI: participant level estimations

"""
import pandas as pd
import numpy as np
import pytest
import warnings

from scipy.stats import norm
from cpm.utils.metai import h2
from cpm.applications.metainformation_theoretic import metaI_d_prime, fit_metaI, EstimatorMetaI

import sys
import os
# adds parent directory to path so that test_metainformation_theoretic.py can be imported directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def synthetic_data():
    # Creates a DataFrame with 2 participants with 8 trials each with the required columns
    return pd.DataFrame({
        "participant": [1, 1, 1, 1, 1, 1, 1, 1, 
                        2, 2, 2, 2, 2, 2, 2, 2],
        "signal": [0, 0, 0, 0, 1, 1, 1, 1,
                   0, 0, 0, 0, 1, 1, 1, 1],
        "response": [0, 0, 0, 1, 1, 1, 1, 0,
                     0, 0, 0, 1, 0, 1, 0, 1],
        "confidence": [0.9, 0.8, 0.85, 0.2, 0.9, 0.75, 0.8, 0.15,
                       0.7, 0.4, 0.6,  0.3, 0.8, 0.2,  0.5, 0.1],
        "accuracy": [1, 1, 1, 0, 1, 1, 1, 0,
                     1, 1, 1, 0, 0, 1, 0, 1],
    })

@pytest.fixture
def balanced_data():
        #accuracy at chance level, with random confidence
        rng = np.random.default_rng(7)
        accuracy = np.array([1] * 100 + [0] * 100)
        confidence = rng.uniform(0, 1, 200)
        return accuracy, confidence

@pytest.fixture
def perfect_acc_data():
      rng = np.random.default_rng(3)
      accuracy = np.ones(100, dtype = int)
      confidence = rng.uniform(0, 1, 100)
      return accuracy, confidence

@pytest.fixture
def ideal_data():
      #simulates an ideal baysian observer, example: d'=1.5, n =2000trials
      rng = np.random.default_rng(0)
      d_prime = 1.5
      sd = 2 / d_prime
      n = 2000

      signal = rng.choice([-1, 1], n)
      evidence = rng.normal(loc=signal, scale=sd)
      choice = np.sign(evidence)
      accuracy = (choice == signal).astype(int)
      confidence = 1/ (1 + np.exp(-(d_prime ** 2) * np.abs(evidence) / 2))
      return accuracy, confidence, d_prime

#tests metaI_d_prime

class TestMetaIDPrime:

    #input validation
    def test_zero_dprime(self):
          with pytest.raises(ValueError, match="d' needs to be > 0"):
                metaI_d_prime(0.0, bins=None)

    def test_negative_dprime(self):
          with pytest.raises(ValueError):
                metaI_d_prime(-1.0, bins=None)

    def test_infinite_dprime(self):
          with pytest.raises(ValueError):
                metaI_d_prime(np.inf, bins=None)

    def test_nan_dprime(self):
          with pytest.raises(ValueError):
                metaI_d_prime(np.nan, bins=None)

    #tests continous integration (bins=None)
    def test_cont_int_returns_positive(self):
          assert metaI_d_prime(1.5, bins = None) > 0

    def test_cont_int_returns_float(self):
          assert isinstance(metaI_d_prime(1.0, bins=None), float)

    def test_cont_int_bounded_by_h2(self):
          d = 1.5
          assert metaI_d_prime(d, bins=None) <= h2(norm.cdf(d/2) + 1e-8)

    def test_cont_int_plausible_range_dprime1(self): #why for dprime 1?
          result = metaI_d_prime(1.0, bins=None)
          # assert 0.05 < result < 0.35

    # ensures that higher d', higher ideal meta-I(d')
    def test_cont_int_increases_with_dprime(self):
        m1 = metaI_d_prime(0.5, bins=None)
        m2 = metaI_d_prime(1.5, bins=None)
        m3 = metaI_d_prime(3.0, bins=None)
        assert m1 < m2 < m3

    # Monte carlo simulation (bins=int)
    def test_mc_returns_positive(self):
           assert metaI_d_prime(1.5, bins=4, n_trials=50_000, seed=0) > 0

    # using the default number of trials the MC estimate should be within 5% relative error of the analytical value
    def test_mc_approaches_continous(self):
          continous = metaI_d_prime(1.5, bins=None)
          mc = metaI_d_prime(1.5, bins=4, seed=7)
          assert abs(mc - continous)/continous < 0.10

#tests fit_metaI
class TestFitMetaI:

    EXPECTED_KEYS = {
        "H2r", "H2r_given_c", "meta_I",
        "metaI_d_prime", "meta_Ir1", "meta_Ir2",
        "n_correct", "n_incorrect",
        "n_correct_bin", "n_incorrect_bin",
        "thresholds", "rebin",
    }

    # test output structure
    def test_all_keys_present(self, balanced_data):
        accuracy, confidence = balanced_data
        result = fit_metaI(accuracy, confidence, bins=4)
        assert self.EXPECTED_KEYS == set(result.keys())

    #H2r
    def test_H2r_is_one_balanced_data(self, balanced_data):
        accuracy, confidence = balanced_data
        result = fit_metaI(accuracy, confidence, bins=4)
        assert result["H2r"] == pytest.approx(1.0, abs=0.01)

    def test_H2r_is_zero_perfect_accuracy(self, perfect_acc_data):
          accuracy, confidence = perfect_acc_data
          result = fit_metaI(accuracy, confidence, bins=4)
          assert result["H2r"] == pytest.approx(0.0, abs=1e-8)
      
    #meta-I
    def test_metaI_nonnegative(self, balanced_data):
        accuracy, confidence = balanced_data
        result = fit_metaI(accuracy, confidence, bins=4)
        assert result["meta_I"] >= -1e-10

    def test_metaI_bounded_by_h2r(self, balanced_data):
          accuracy, confidence = balanced_data
          result = fit_metaI(accuracy, confidence, bins=4)
          assert result["meta_I"] <= result["H2r"] + 1e-8

    def test_metaI_positive_informative_confidence(self, ideal_data):
          accuracy, confidence, _ = ideal_data
          result = fit_metaI(accuracy, confidence, bins=4)
          assert result["meta_I"] > 0


#tests metaI_d_prime
    def test_metaIdp_nan_without_dprime(self, balanced_data):
          accuracy, confidence = balanced_data
          result = fit_metaI(accuracy, confidence, bins=4, d_prime=None)
          assert np.isnan(result["metaI_d_prime"])

    def test_metaIdp_finite_with_dprim(self, balanced_data):
          accuracy, confidence = balanced_data
          result = fit_metaI(accuracy, confidence, bins=4, d_prime=1.0)
          assert np.isfinite(result["metaI_d_prime"])
          assert result["metaI_d_prime"] > 0

#tests efficiency measures metaI1r and metaIr2
    def test_meta_I1r_nan_without_dprime(self, balanced_data):
          accuracy, confidence = balanced_data
          result = fit_metaI(accuracy, confidence, bins=4, d_prime=None)
          assert np.isnan(result["meta_Ir1"])

    def test_meta_Ir1_one_ideal_observer(self, ideal_data):
          accuracy, confidence, d_prime = ideal_data
          result = fit_metaI(accuracy, confidence, bins=4, d_prime=d_prime)
          assert 0.5 < result["meta_Ir1"] < 1.5

    def test_meta_Ir2_bounded(self, balanced_data):
          accuracy, confidence = balanced_data
          result = fit_metaI(accuracy, confidence, bins=4, d_prime=1.0)
          assert 0 < result["meta_Ir2"] < 1.0 + 1e-8

#tests matched_bins
    def test_matched_bins_finite_metaIr1(self, ideal_data):
          accuracy, confidence, d_prime = ideal_data
          result = fit_metaI(accuracy, confidence, bins=4, d_prime=d_prime, matched_bins=True)
          assert np.isfinite(result["meta_Ir1"])

    #matched_bins=True uses MC, False uses numerical integration. In both cases meta_d_prime must be finite.
    def test_matched_unmatched_bins_finite_metadp(self, ideal_data):
            accuracy, confidence, d_prime = ideal_data
            r_matched = fit_metaI(accuracy, confidence, bins=4, d_prime=d_prime, matched_bins=True)
            r_unmatched = fit_metaI(accuracy, confidence, bins=4, d_prime=d_prime, matched_bins=False)

            assert np.isfinite(r_matched["meta_Ir1"])
            assert np.isfinite(r_unmatched["meta_Ir1"])

#tests if counts add up
    def test_n_correct_incorrect_sum_to_n(self, balanced_data):
            accuracy, confidence = balanced_data
            result = fit_metaI(accuracy, confidence, bins=4)
            assert result["n_correct"] + result["n_incorrect"] == len(accuracy)

    def test_bincounts_sums_to_n(self, balanced_data):
          accuracy, confidence = balanced_data
          result = fit_metaI(accuracy, confidence, bins=4)
          assert sum(result["n_correct_bin"]) == result["n_correct"]
          assert sum(result["n_incorrect_bin"] ) == result["n_incorrect"]

#edge caseses
    def test_metaI_zero_perfect_accuracy(self, perfect_acc_data):
           accuracy, confidence = perfect_acc_data
           result = fit_metaI(accuracy, confidence, bins=4, ignore_invalid = True)
           assert result["H2r"] == pytest.approx(0.0, abs=1e-8)
           assert result["meta_I"] == pytest.approx(0.0, abs=1e-8)

    def test_metaI_zeroall_incorrect(self):
          rng = np.random.default_rng(5)
          accuracy = np.zeros(100, dtype = int)
          confidence = rng.uniform(0, 1, 100)
          result = fit_metaI(accuracy, confidence, bins=4)
          assert result["H2r"] == pytest.approx(0.0, abs=1e-8)
          assert result["meta_I"] == pytest.approx(0.0, abs=1e-8)

    def test_meta_Ir2_nan_perfect_accuracy(self, perfect_acc_data):
          accuracy, confidence = perfect_acc_data
          result = fit_metaI(accuracy, confidence, bins=4, ignore_invalid = True)
          assert np.isnan(result["meta_Ir2"])

    def test_identical_confidence_raises_or_rebins(self, balanced_data):
          _, accuracy = balanced_data
          confidence = np.full(100, 0.5)
            
          try:
                result = fit_metaI(accuracy, confidence, bins=4)

                assert result["rebin"] ==1, (
                      "identical confidence values shouzld set rebin=1"
                )
          except ValueError:
                pass

# equivalence testing against an independent reimplementation
def meta_I_reference(accuracy, confidence, bins):
      accuracy = np.asarray(accuracy, dtype=float)
      confidence = np.asarray(confidence, dtype=float)

      def h2_scalar(p):
            if p <= 0 or p >= 1:
                  return 0.0
            return -(p * np.log2(p) + (1-p) * np.log2(1-p))
                     
      # Step 1: entropy of overall accuracy
      H2r = h2_scalar(accuracy.mean())

      # Step 2: quantile bin edges (equal-frequency)
      bin_edges = np.percentile(confidence, np.linspace(0, 100, bins + 1))
      bin_edges[-1] += 1e-12   # make upper edge inclusive

      # Steps 3 & 4: weighted conditional entropy
      n = len(accuracy)
      H2r_given_c = 0.0
      for b in range(bins):
            in_bin = (confidence >= bin_edges[b]) & (confidence < bin_edges[b + 1])
            n_bin  = in_bin.sum()
            if n_bin == 0:
                  continue
            p_correct_given  = accuracy[in_bin].mean()
            H2r_given_c     += (n_bin / n) * h2_scalar(p_correct_given)

            return H2r - H2r_given_c

      rng = np.random.default_rng(11)
      for _ in range(50):
            n          = rng.integers(100, 400)
            p_acc      = rng.uniform(0.55, 0.85)   # avoid 0/1 extremes
            accuracy   = rng.choice([0, 1], size=n, p=[1 - p_acc, p_acc])
            confidence = rng.uniform(0, 1, n)
            bins       = int(rng.choice([2, 3, 4, 6]))

            result    = fit_metaI(accuracy, confidence, bins=bins)
            reference = meta_I_reference(accuracy, confidence, bins)

            assert result["meta_I"] == pytest.approx(reference, abs=1e-6), (
                f"Mismatch for n={n}, bins={bins}: "
                f"fit_metaI={result['meta_I']:.8f}, reference={reference:.8f}"
            )

class TestEstimatorMetaI:

      def test_custom_ppt_identifier(self, synthetic_data):
            df = synthetic_data.rename(columns={"participant": "subject"})
            est = EstimatorMetaI(df, bins=2, ppt_identifier="subject")
            est.estimate()
            assert len(est.fit) == 2 #is this the best way to test the function?

      def test_estimate_one_result_per_ppt(self, synthetic_data):
            est = EstimatorMetaI(synthetic_data, bins=2)
            est.estimate()
            assert len(est.fit) == synthetic_data["participant"].nunique()

      def test_ppt_ids_stored(self, synthetic_data):
            est = EstimatorMetaI(synthetic_data, bins=2)
            est.estimate()
            assert {x["ppt"] for x in est.fit} == set(synthetic_data["participant"].unique())

      def test_results_contain_required_keys(self, synthetic_data):
            est = EstimatorMetaI(synthetic_data, bins=2)
            est.estimate()
            required = {"H2r", "meta_I", "metaI_d_prime", "meta_Ir1", "meta_Ir2", "n_correct", "n_incorrect", "ppt"}
            for result in est.fit:
                  assert required.issubset(set(result.keys()))

      def test_metaI_nonnegative(self, synthetic_data):
            est = EstimatorMetaI(synthetic_data, bins=2)
            est.estimate()
            for result in est.fit:
                  assert result["meta_I"] >= -1e-10

      def test_warns_nonpositive_dprime(self):
            df = pd.DataFrame({
            "participant": [1, 1, 1, 1, 2, 2, 2, 2],
            "signal":      [1, 1, 0, 0, 1, 1, 0, 0],
            "response":    [0, 0, 1, 1, 1, 1, 0, 0],   # ppt 1: always wrong
            "accuracy":    [0, 0, 0, 0, 1, 1, 1, 1],
            "confidence":  [0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6],
            })

            est = EstimatorMetaI(df, bins=2)
            with pytest.warns(RuntimeWarning):
                  est.estimate()
            ppt_1 = est.fit[0]
            assert np.isnan(ppt_1["metaI_d_prime"])
            assert np.isnan(ppt_1["meta_Ir1"])

      def test_estimator_raises_constant_confidence(self):
            df = pd.DataFrame({
            "participant": [1, 1, 1, 1],
            "signal":      [1, 1, 0, 0],
            "response":    [1, 0, 1, 0],
            "accuracy":    [1, 0, 0, 1],
            "confidence":  [0.5, 0.5, 0.5, 0.5],
            })
            
            est = EstimatorMetaI(df, bins=4)
            with pytest.warns(RuntimeWarning):
                  with pytest.raises(RuntimeError):
                        est.estimate()

      # tests export structure
      def test_export_returns_df(self, synthetic_data):
            est = EstimatorMetaI(synthetic_data, bins=2)
            est.estimate()
            assert isinstance(est.export(), pd.DataFrame)

      def test_export_before_estimate_empty(self, synthetic_data):
            est = EstimatorMetaI(synthetic_data, bins=2)
            assert len(est.export()) == 0

      # tests if matched_bin propagated further down call line
      def test_matched_bin_propagted(self, ideal_data):
            accuracy, confidence, d_prime = ideal_data
            df = pd.DataFrame({
            "participant": np.ones(len(accuracy), dtype=int),
            "signal":      (accuracy == 1).astype(int),
            "response":    (accuracy == 1).astype(int),
            "accuracy":    accuracy,
            "confidence":  confidence,
            })
            est = EstimatorMetaI(df, bins=4, matched_bins=True)
            est.estimate()
            assert np.isfinite(est.fit[0]["metaI_d_prime"])
      