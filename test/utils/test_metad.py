import warnings

import numpy as np
import pandas as pd
import pytest

from cpm.utils.metad import count_trials

STIM = [0, 1, 0, 0, 1, 1, 1, 1]
ACC = [0, 1, 1, 1, 0, 0, 1, 1]
CONF = [1, 2, 3, 4, 4, 3, 2, 1]
RESP = [1, 1, 0, 0, 0, 0, 1, 1]  # inferred from STIM and ACC
EXPECTED_S1 = [1, 1, 0, 0, 1, 0, 0, 0]
EXPECTED_S2 = [1, 1, 0, 0, 1, 2, 0, 0]


def assert_expected(result):
    nR_S1, nR_S2 = result
    np.testing.assert_array_equal(nR_S1, EXPECTED_S1)
    np.testing.assert_array_equal(nR_S2, EXPECTED_S2)


def test_responses_only_dataframe():
    data = pd.DataFrame({"Stimuli": STIM, "Responses": RESP, "Confidence": CONF})
    assert_expected(count_trials(data, nRatings=4))


def test_infers_responses_from_accuracy_dataframe():
    data = pd.DataFrame({"Stimuli": STIM, "Accuracy": ACC, "Confidence": CONF})
    assert_expected(count_trials(data, nRatings=4))


def test_infers_responses_from_accuracy_arrays():
    assert_expected(
        count_trials(stimuli=STIM, accuracy=ACC, confidence=CONF, nRatings=4)
    )


def test_responses_arrays():
    assert_expected(
        count_trials(
            stimuli=STIM, responses=RESP, accuracy=None, confidence=CONF, nRatings=4
        )
    )


def test_neither_responses_nor_accuracy_raises():
    data = pd.DataFrame({"Stimuli": STIM, "Confidence": CONF})
    with pytest.raises(ValueError, match="responses.*accuracy"):
        count_trials(data, nRatings=4)


def test_invalid_accuracy_trials_dropped():
    data = pd.DataFrame(
        {"Stimuli": STIM + [0], "Accuracy": ACC + [2], "Confidence": CONF + [1]}
    )
    assert_expected(count_trials(data, nRatings=4))


def test_nan_accuracy_trials_dropped():
    data = pd.DataFrame(
        {"Stimuli": STIM + [1], "Accuracy": ACC + [np.nan], "Confidence": CONF + [2]}
    )
    assert_expected(count_trials(data, nRatings=4))


def test_boolean_accuracy():
    data = pd.DataFrame(
        {"Stimuli": STIM, "Accuracy": [bool(a) for a in ACC], "Confidence": CONF}
    )
    assert_expected(count_trials(data, nRatings=4))


def test_no_valid_trials_raises():
    data = pd.DataFrame(
        {"Stimuli": [s + 2 for s in STIM], "Accuracy": ACC, "Confidence": CONF}
    )
    with pytest.raises(ValueError, match="no valid trials"), pytest.warns(
        UserWarning, match="outside"
    ):
        count_trials(data, nRatings=4)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="length"):
        count_trials(stimuli=STIM, accuracy=ACC[:-1], confidence=CONF, nRatings=4)


def test_padding_default_amount():
    data = pd.DataFrame({"Stimuli": STIM, "Responses": RESP, "Confidence": CONF})
    nR_S1, nR_S2 = count_trials(data, nRatings=4, padding=True)
    np.testing.assert_allclose(nR_S1, np.array(EXPECTED_S1) + 1 / 8)
    np.testing.assert_allclose(nR_S2, np.array(EXPECTED_S2) + 1 / 8)


def test_non_dataframe_data_raises():
    with pytest.raises(ValueError, match="DataFrame"):
        count_trials(data=[1, 2, 3])


def test_consistent_responses_and_accuracy_match_responses_only():
    data = pd.DataFrame(
        {"Stimuli": STIM, "Responses": RESP, "Accuracy": ACC, "Confidence": CONF}
    )
    assert_expected(count_trials(data, nRatings=4))


def test_inconsistent_responses_and_accuracy_raises():
    accuracy = ACC.copy()
    accuracy[2] = 0
    data = pd.DataFrame(
        {"Stimuli": STIM, "Responses": RESP, "Accuracy": accuracy, "Confidence": CONF}
    )
    with pytest.raises(ValueError, match="inconsistent") as error:
        count_trials(data, nRatings=4)
    assert "1 trial" in str(error.value)
    assert "[2]" in str(error.value)


def test_inconsistency_on_dropped_trial_ignored():
    data = pd.DataFrame(
        {
            "Stimuli": STIM + [1],
            "Responses": RESP + [1],
            "Accuracy": ACC + [0],
            "Confidence": CONF + [9],
        }
    )
    assert_expected(count_trials(data, nRatings=4))


def test_stimuli_outside_0_1_warn():
    data = pd.DataFrame(
        {"Stimuli": [s + 1 for s in STIM], "Accuracy": ACC, "Confidence": CONF}
    )
    with pytest.warns(UserWarning, match=r"5 trial\(s\).*outside \{0, 1\}"):
        count_trials(data, nRatings=4)


def test_valid_and_missing_stimuli_do_not_warn():
    data = pd.DataFrame(
        {"Stimuli": STIM + [np.nan], "Accuracy": ACC + [1], "Confidence": CONF + [1]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert_expected(count_trials(data, nRatings=4))


def test_non_integer_confidence_raises():
    data = pd.DataFrame({"Stimuli": STIM, "Accuracy": ACC, "Confidence": [2.5] * 8})
    with pytest.raises(ValueError, match="no valid trials"):
        count_trials(data, nRatings=4)


def test_no_valid_trials_message_names_every_rule():
    # The message should list every rule the Notes section documents (#107).
    data = pd.DataFrame({"Stimuli": STIM, "Accuracy": [2] * 8, "Confidence": CONF})
    with pytest.raises(ValueError) as excinfo:
        count_trials(data, nRatings=4)
    message = str(excinfo.value)
    assert "`accuracy`" in message
    assert "whole number in [1, 4]" in message
