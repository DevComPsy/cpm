import numpy as np
import pandas as pd
import pytest

from cpm.brainexplorer.bandits import MilkyWay


def _game_rows(
    user_id,
    trial_type,
    correct,
    rep=None,
    run=1,
    outcomes=None,
    date="2025-02-20 09:00:00.000",
):
    n = len(correct)
    rep = rep if rep is not None else [1, 0] * (n // 2) + [1] * (n % 2)
    outcomes = outcomes if outcomes is not None else [10] * n
    return [
        {
            "userID": user_id,
            "trial_type": trial_type,
            "run": run,
            "date": date,
            "correct": c,
            "outchosen": outcome,
            "obt_min_forg": outcome - 5,
            "rep": r,
            "WSLS_v1": c,
            "WSLS_v2": 1 - c,
        }
        for c, r, outcome in zip(correct, rep, outcomes)
    ]


def _both_games(user_id, **kwargs):
    return _game_rows(user_id, "reward", [1, 1, 0, 1], **kwargs) + _game_rows(user_id, "punish", [1, 0, 0, 1], **kwargs)


def test_milkyway_splits_games_and_keeps_the_first_attempt():
    rows = _game_rows("u1", "reward", [1, 1], run=1)
    rows += _game_rows("u1", "reward", [0, 0], run=2)
    # the first Pirate Market attempt is not necessarily run 1
    rows += _game_rows("u1", "punish", [1, 0], run=2)
    rows += _game_rows("u1", "punish", [0, 0], run=3)
    milky_way = MilkyWay(pd.DataFrame(rows))
    assert set(milky_way.MW_data["run"]) == {1}
    assert set(milky_way.PM_data["run"]) == {2}


def test_milkyway_metrics_computes_expected_values():
    rows = _game_rows("u1", "reward", [1, 1, 0, 1], rep=[1, 1, 0, 0], outcomes=[10, 20, 30, 40])
    rows += _game_rows("u1", "punish", [1, 0, 0, 0], rep=[1, 0, 0, 0], outcomes=[-10, -20, -30, -40])
    milky_way = MilkyWay(pd.DataFrame(rows))
    results_MW, results_PM = milky_way.metrics()

    mw = results_MW.iloc[0]
    assert mw["trial_type"] == "Milky Way"
    assert mw["n_trials"] == 4
    assert np.isclose(mw["accuracy_MW"], 0.75)
    assert np.isclose(mw["mean_outcome_MW"], 25)
    assert np.isclose(mw["reward_diff_obt_forg_MW"], 20)
    assert np.isclose(mw["prop_same_choice_MW"], 0.5)
    assert np.isclose(mw["prop_WSLS_1_MW"], 0.75)
    assert np.isclose(mw["prop_WSLS_2_MW"], 0.25)

    pm = results_PM.iloc[0]
    assert pm["trial_type"] == "Pirate Market"
    assert np.isclose(pm["accuracy_PM"], 0.25)
    assert np.isclose(pm["mean_outcome_PM"], -25)
    assert np.isclose(pm["prop_same_choice_PM"], 0.25)
    assert results_MW is milky_way.results_MW and results_PM is milky_way.results_PM


def test_milkyway_results_contain_the_session_time():
    milky_way = MilkyWay(pd.DataFrame(_both_games("u1", date="2025-02-20 03:30:00.000")))
    for results in milky_way.metrics():
        row = results.iloc[0]
        assert row["day_of_week"] == "Thursday"
        assert str(row["time"]) == "03:30:00"
        assert row["time_of_day"] == "night"


def test_milkyway_difference_metrics_only_for_participants_who_played_both():
    rows = _game_rows("both", "reward", [1, 1, 1, 1], outcomes=[10, 10, 10, 10])
    rows += _game_rows("both", "punish", [1, 0, 1, 0], outcomes=[-10, -10, -10, -10])
    rows += _game_rows("reward_only", "reward", [1, 0, 1, 0])
    milky_way = MilkyWay(pd.DataFrame(rows))
    differences = milky_way.difference_metrics()  # also runs metrics

    assert list(differences["userID"]) == ["both"]
    row = differences.iloc[0]
    assert np.isclose(row["accuracy_diff"], 0.5)
    assert np.isclose(row["outcome_diff"], 20)
    assert np.isclose(row["reward_diff_obt_forg_diff"], 20)
    assert np.isclose(row["prop_WSLS_1_diff"], 0.5)
    assert np.isclose(row["prop_WSLS_2_diff"], -0.5)


def test_milkyway_difference_metrics_without_pirate_market():
    milky_way = MilkyWay(pd.DataFrame(_game_rows("u1", "reward", [1, 0])))
    differences = milky_way.difference_metrics()
    assert differences.empty
    assert "accuracy_diff" in differences.columns


def test_milkyway_metrics_can_be_run_twice():
    milky_way = MilkyWay(pd.DataFrame(_both_games("u1")))
    first_MW, first_PM = (results.copy() for results in milky_way.metrics())
    second_MW, second_PM = milky_way.metrics()
    pd.testing.assert_frame_equal(first_MW, second_MW)
    pd.testing.assert_frame_equal(first_PM, second_PM)
    first_diff = milky_way.difference_metrics().copy()
    pd.testing.assert_frame_equal(first_diff, milky_way.difference_metrics())


def test_milkyway_clean_data_excludes_invalid_participants():
    rows = _both_games("keep")
    # same choice on at least 95% of Milky Way trials
    rows += _game_rows("drop_MW_same", "reward", [1] * 20, rep=[1] * 20)
    rows += _game_rows("drop_MW_same", "punish", [1, 0, 0, 1])
    # more than 72 Pirate Market trials
    rows += _game_rows("drop_PM_trials", "reward", [1, 1, 0, 1])
    rows += _game_rows("drop_PM_trials", "punish", [1, 0] * 37)
    # missing Milky Way accuracy
    rows += _game_rows("drop_MW_accuracy", "reward", [np.nan, np.nan])
    rows += _game_rows("drop_MW_accuracy", "punish", [1, 0, 0, 1])

    milky_way = MilkyWay(pd.DataFrame(rows))
    cleaned_MW, cleaned_PM, cleaned_diff = milky_way.clean_data()  # also runs metrics and difference_metrics

    assert set(cleaned_MW["userID"]) == {"keep", "drop_PM_trials"}
    assert set(cleaned_PM["userID"]) == {"keep", "drop_MW_same", "drop_MW_accuracy"}
    assert set(cleaned_diff["userID"]) == {"keep"}
    assert milky_way.deleted_participants_MW == 2
    assert milky_way.deleted_participants_PM == 1
    assert milky_way.deleted_participants_diff == 3


def test_milkyway_requires_data_and_columns():
    with pytest.raises(ValueError, match="needs data"):
        MilkyWay()
    data = pd.DataFrame(_both_games("u1")).drop(columns=["obt_min_forg"])
    with pytest.raises(KeyError, match="obt_min_forg"):
        MilkyWay(data)


def test_milkyway_codebook_describes_every_metric():
    milky_way = MilkyWay(pd.DataFrame(_both_games("u1")))
    results_MW, results_PM = milky_way.metrics()
    differences = milky_way.difference_metrics()
    columns = set(results_MW.columns) | set(results_PM.columns) | set(differences.columns)
    assert columns == set(milky_way.get_codebook())


def test_milkyway_reads_csv(tmp_path):
    filepath = tmp_path / "milky_way.csv"
    pd.DataFrame(_both_games("u1")).to_csv(filepath, index=False)
    results_MW, results_PM = MilkyWay(filepath).metrics()
    assert len(results_MW) == 1 and len(results_PM) == 1
