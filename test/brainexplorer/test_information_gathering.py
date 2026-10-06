import numpy as np
import pandas as pd
import pytest

from cpm.brainexplorer.information_gathering import TreasureHunt


def _participant_rows(
    user_id,
    draws,
    chose_current_evidence,
    outcomes=None,
    confidences=None,
    run=1,
    date="2025-02-20 09:00:00.000",
):
    n = len(draws)
    outcomes = outcomes if outcomes is not None else [10] * n
    confidences = confidences if confidences is not None else [50] * n
    return [
        {
            "userID": user_id,
            "date": date,
            "run": run,
            "outcome": outcome,
            "confidence": confidence,
            "RT": 1000,
            "confidenceRT": 400 + 10 * i,
            "draws": draw,
            "choseCurEv": chose,
            "median_diffRT": 200 + 10 * i,
            "ev": "[1 2 -1]",
        }
        for i, (draw, chose, outcome, confidence) in enumerate(zip(draws, chose_current_evidence, outcomes, confidences))
    ]


def _keep_rows(user_id="keep", **kwargs):
    return _participant_rows(user_id, draws=[3, 5, 7, 9, 11], chose_current_evidence=[1, 1, 1, 1, 1], **kwargs)


def test_treasurehunt_keeps_only_the_first_attempt():
    rows = _keep_rows() + _keep_rows(run=2)
    hunt = TreasureHunt(pd.DataFrame(rows))
    assert len(hunt.data) == 5
    assert (hunt.data["run"] == 1).all()


def test_treasurehunt_metrics_computes_expected_values():
    rows = _participant_rows(
        "u1",
        draws=[2, 4, 6, 8],
        chose_current_evidence=[1, 0, 1, 1],
        outcomes=[10, 20, 30, 40],
        confidences=[20, 40, 60, 80],
        date="2025-02-20 14:30:00.000",
    )
    hunt = TreasureHunt(pd.DataFrame(rows))
    result = hunt.metrics().iloc[0]

    assert result["n_trials"] == 4
    assert result["day_of_week"] == "Thursday"
    assert result["time_of_day"] == "afternoon"
    assert np.isclose(result["mean_points"], 25)
    assert np.isclose(result["accuracy"], 0.75)
    assert np.isclose(result["mean_confidence"], 50)
    assert np.isclose(result["sd_confidence"], np.std([20, 40, 60, 80]))
    assert np.isclose(result["mean_n_draws"], 5)
    assert np.isclose(result["sd_n_draws"], np.std([2, 4, 6, 8]))
    assert np.isclose(result["median_RT_between_actions"], 215)  # median([200, 210, 220, 230])
    assert np.isclose(result["median_confidenceRT"], 415)  # median([400, 410, 420, 430])
    assert result["unique_draws"] == 4


@pytest.mark.parametrize(
    "date, expected",
    [
        ("2025-02-20 02:00:00.000", "night"),
        ("2025-02-20 06:00:00.000", "morning"),
        ("2025-02-20 12:00:00.000", "afternoon"),
        ("2025-02-20 23:59:00.000", "evening"),
    ],
)
def test_treasurehunt_time_of_day(date, expected):
    hunt = TreasureHunt(pd.DataFrame(_keep_rows(date=date)))
    assert hunt.metrics().iloc[0]["time_of_day"] == expected


def test_treasurehunt_median_confidence_rt_ignores_missing_values():
    rows = _keep_rows()
    rows[0]["confidenceRT"] = np.nan
    result = TreasureHunt(pd.DataFrame(rows)).metrics().iloc[0]
    assert np.isclose(result["median_confidenceRT"], 425)  # median([410, 420, 430, 440])


@pytest.mark.parametrize("evidence", ["[1 2 -1]", "[1  2 -1]", "[1, 2, -1]", " [1 2 -1] "])
def test_treasurehunt_parses_evidence(evidence):
    rows = _keep_rows()
    for row in rows:
        row["ev"] = evidence
    hunt = TreasureHunt(pd.DataFrame(rows))
    assert hunt.data["ev"].iloc[0] == [1, 2, -1]
    hunt.metrics()
    hunt.metrics()  # the parsed evidence must survive a second run
    assert hunt.data["ev"].iloc[0] == [1, 2, -1]


def test_treasurehunt_evidence_column_is_optional():
    data = pd.DataFrame(_keep_rows()).drop(columns=["ev"])
    assert len(TreasureHunt(data).metrics()) == 1


def test_treasurehunt_requires_data_and_columns():
    with pytest.raises(ValueError, match="needs data"):
        TreasureHunt()
    data = pd.DataFrame(_keep_rows()).drop(columns=["median_diffRT"])
    with pytest.raises(KeyError, match="median_diffRT"):
        TreasureHunt(data)


def test_treasurehunt_metrics_can_be_run_twice():
    hunt = TreasureHunt(pd.DataFrame(_keep_rows()))
    first = hunt.metrics().copy()
    pd.testing.assert_frame_equal(first, hunt.metrics())


def test_treasurehunt_clean_data_excludes_invalid_participants():
    rows = _keep_rows()
    # mean number of draws < 2
    rows += _participant_rows("drop_few_draws", draws=[1, 1, 1, 2, 3], chose_current_evidence=[1] * 5)
    # mean number of draws > 23
    rows += _participant_rows("drop_many_draws", draws=[22, 24, 26, 28, 30], chose_current_evidence=[1] * 5)
    # 20% of choices not in line with the evidence
    rows += _participant_rows("drop_accuracy", draws=[3, 5, 7, 9, 11], chose_current_evidence=[1, 1, 1, 1, 0])
    # fewer than 3 unique numbers of draws
    rows += _participant_rows("drop_unique", draws=[4, 4, 6, 6, 6], chose_current_evidence=[1] * 5)

    hunt = TreasureHunt(pd.DataFrame(rows))
    cleaned = hunt.clean_data()  # also runs metrics
    assert set(cleaned["userID"]) == {"keep"}
    assert hunt.deleted_participants == 4


def test_treasurehunt_codebook_describes_every_metric():
    hunt = TreasureHunt(pd.DataFrame(_keep_rows()))
    assert set(hunt.metrics().columns) == set(hunt.get_codebook())


def test_treasurehunt_reads_csv(tmp_path):
    filepath = tmp_path / "treasure_hunt.csv"
    pd.DataFrame(_keep_rows()).to_csv(filepath, index=False)
    hunt = TreasureHunt(str(filepath))
    assert hunt.data["ev"].iloc[0] == [1, 2, -1]
    assert len(hunt.metrics()) == 1
