import numpy as np
import pandas as pd
import pytest

from cpm.brainexplorer.risky_decision_making import Scavenger


def _trial(user_id, chosen, loss=False, ambiguous=False, ev_diff=1.0, rt=500, outcome=10, **extra):
    row = {
        "userID": user_id,
        "run": 1,
        "date": "2025-01-10 15:00:00.000",
        "RT": rt,
        "chosen": chosen,
        "ambiguousTrial": int(ambiguous),
        "abovechance": 1,
        "EV_diff_chosen": ev_diff,
        "safe_EV": -5 if loss else 5,
        "outcome": outcome,
        "nbcorrect_gain": np.nan,
        "resp_rep": 0,
        "resp_rep_sr": 0,
        "config_id": 1,
    }
    row.update(extra)
    return row


def _example(user_id="u1"):
    """Five trials, worked out by hand in the tests below; chosen is 1 for safe and 2 for risky."""
    return [
        _trial(user_id, chosen=2, ev_diff=5, rt=300, outcome=10),
        _trial(user_id, chosen=1, ambiguous=True, ev_diff=-2, rt=600, outcome=5),
        _trial(user_id, chosen=1, loss=True, ev_diff=0, rt=900, outcome=-5),
        _trial(user_id, chosen=2, loss=True, ambiguous=True, ev_diff=3, rt=400, outcome=0),
        _trial(user_id, chosen=2, ev_diff=1, rt=500, outcome=20),
    ]


def _keep_rows(user_id="keep"):
    return [_trial(user_id, chosen=1 + i % 2, loss=i % 4 >= 2, ambiguous=i % 3 == 0, resp_rep=i % 2) for i in range(12)]


def test_scavenger_recodes_safe_and_risky_choices():
    scavenger = Scavenger(pd.DataFrame(_example()))
    assert list(scavenger.data["risky"]) == [1, 0, 0, 1, 1]


@pytest.mark.parametrize("codes", [[0, 1], [1, 3]])
def test_scavenger_rejects_other_codings(codes):
    rows = [_trial("u1", chosen=code) for code in codes]
    with pytest.raises(ValueError, match="1 \\(safe\\) or 2 \\(risky\\)"):
        Scavenger(pd.DataFrame(rows))


def test_scavenger_trial_level_exclusions():
    rows = _example()
    rows.append(_trial("u1", chosen=2, rt=100))  # too fast
    rows.append(_trial("u1", chosen=2, rt=12000))  # too slow
    rows.append(_trial("u1", chosen=2, run=2))  # second attempt
    scavenger = Scavenger(pd.DataFrame(rows))
    assert len(scavenger.data) == 5


def test_scavenger_choice_metrics():
    result = Scavenger(pd.DataFrame(_example())).metrics().iloc[0]

    assert result["n_trials"] == 5
    assert result["time_of_day"] == "afternoon"
    assert np.isclose(result["risky_choices"], 0.6)
    assert np.isclose(result["win_risk"], 2 / 3)
    assert np.isclose(result["loss_risk"], 0.5)
    assert np.isclose(result["diff_loss_win_risk"], 0.5 - 2 / 3)
    assert np.isclose(result["risk_Ambg"], 0.5)
    assert np.isclose(result["risk_NoAmbg"], 2 / 3)
    assert np.isclose(result["win_risk_abg"], 0)
    assert np.isclose(result["win_risk_NoAbg"], 1)
    assert np.isclose(result["loss_risk_abg"], 1)
    assert np.isclose(result["loss_risk_NoAbg"], 0)
    assert result["choice_stickiness"] == 2  # chosen 2, 1, 1, 2, 2


def test_scavenger_rational_metrics():
    result = Scavenger(pd.DataFrame(_example())).metrics().iloc[0]

    # rational: EV_diff_chosen >= 0, which is all but the second trial
    assert np.isclose(result["rational_all"], 0.8)
    assert np.isclose(result["rational_win"], 2 / 3)
    assert np.isclose(result["rational_loss"], 1)
    assert np.isclose(result["rational_abg"], 0.5)
    assert np.isclose(result["rational_NoAbg"], 1)
    assert np.isclose(result["rational_safe"], 0.5)
    assert np.isclose(result["rational_risky"], 1)
    # safe & win is only the second trial, which was irrational
    assert np.isclose(result["rational_safe_win"], 0)
    assert np.isclose(result["rational_safe_loss"], 1)
    assert np.isclose(result["rational_risky_win"], 1)
    assert np.isclose(result["rational_risky_loss"], 1)
    assert np.isclose(result["diff_rational_safe_win_risky_win"], -1)
    assert np.isclose(result["rational_safe_abg"], 0)
    assert np.isclose(result["rational_safe_NoAbg"], 1)
    assert np.isclose(result["rational_risky_abg"], 1)
    assert np.isclose(result["rational_risky_NoAbg"], 1)


def test_scavenger_rt_and_points_metrics():
    result = Scavenger(pd.DataFrame(_example())).metrics().iloc[0]

    assert np.isclose(result["mean_RT"], 540)
    assert np.isclose(result["median_RT_safe"], 750)
    assert np.isclose(result["median_RT_risky"], 400)
    assert np.isclose(result["median_RT_win"], 500)
    assert np.isclose(result["median_RT_loss"], 650)
    assert np.isclose(result["median_RT_ambg"], 500)
    assert np.isclose(result["median_RT_NoAmbg"], 500)
    assert np.isclose(result["median_RT_loss_ambg"], 400)
    assert np.isclose(result["diff_RT_loss_amg_loss_noAmbg"], 400 - 900)

    assert np.isclose(result["total_rewards"], 30)
    assert np.isclose(result["mean_points"], 6)
    assert np.isclose(result["mean_points_safe"], 0)
    assert np.isclose(result["mean_points_risky"], 10)
    assert np.isclose(result["mean_points_win_NoAmbg"], 15)
    assert np.isclose(result["mean_points_loss_ambg"], 0)


def test_scavenger_quality_metrics():
    rows = _example()
    rows[0].update(nbcorrect_gain=0, resp_rep=1, resp_rep_sr=1, abovechance=0)
    rows[1].update(nbcorrect_gain=1, resp_rep=1)
    result = Scavenger(pd.DataFrame(rows)).metrics().iloc[0]
    assert result["nb_incorrect_gain"] == 1
    assert np.isclose(result["above_chance"], 0.8)
    assert np.isclose(result["chosen_prop_LeftRight"], 0.4)
    assert np.isclose(result["chosen_prop_SafeRisky"], 0.2)


def test_scavenger_metrics_of_missing_conditions_are_nan(recwarn):
    rows = [_trial("u1", chosen=1 + i % 2) for i in range(4)]  # win, non-ambiguous trials only
    result = Scavenger(pd.DataFrame(rows)).metrics().iloc[0]
    assert np.isnan(result["loss_risk"])
    assert np.isnan(result["rational_loss_abg"])
    assert np.isnan(result["median_RT_ambg"])
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_scavenger_metrics_can_be_run_twice():
    scavenger = Scavenger(pd.DataFrame(_example()))
    first = scavenger.metrics().copy()
    pd.testing.assert_frame_equal(first, scavenger.metrics())


def test_scavenger_clean_data_excludes_invalid_participants():
    rows = _keep_rows()
    # failed 2 catch trials
    catch = _keep_rows("drop_catch")
    catch[0]["nbcorrect_gain"] = 0
    catch[1]["nbcorrect_gain"] = 0
    rows += catch
    # repeated the same left/right response on at least 95% of trials
    rows += [{**row, "resp_rep": 1} for row in _keep_rows("drop_left_right")]
    # more than 40 trials
    rows += [_trial("drop_trials", chosen=1 + i % 2) for i in range(41)]
    # the 60-trial version of the game
    rows += [{**row, "config_id": 242} for row in _keep_rows("drop_config")]

    scavenger = Scavenger(pd.DataFrame(rows))
    cleaned = scavenger.clean_data()  # also runs metrics
    assert set(cleaned["userID"]) == {"keep"}
    assert scavenger.deleted_participants == 4


def test_scavenger_clean_data_keeps_one_failed_catch_trial():
    rows = _keep_rows()
    rows[0]["nbcorrect_gain"] = 0
    assert list(Scavenger(pd.DataFrame(rows)).clean_data()["userID"]) == ["keep"]


def test_scavenger_requires_data_and_columns():
    with pytest.raises(ValueError, match="needs data"):
        Scavenger()
    data = pd.DataFrame(_example()).drop(columns=["safe_EV"])
    with pytest.raises(KeyError, match="safe_EV"):
        Scavenger(data)
    # config_id is only needed to clean the data
    scavenger = Scavenger(pd.DataFrame(_example()).drop(columns=["config_id"]))
    assert len(scavenger.metrics()) == 1
    with pytest.raises(KeyError, match="config_id"):
        scavenger.clean_data()


def test_scavenger_codebook_describes_every_metric():
    scavenger = Scavenger(pd.DataFrame(_example()))
    assert set(scavenger.metrics().columns) == set(scavenger.get_codebook())


def test_scavenger_reads_csv(tmp_path):
    filepath = tmp_path / "scavenger.csv"
    pd.DataFrame(_example()).to_csv(filepath, index=False)
    assert np.isclose(Scavenger(str(filepath)).metrics().iloc[0]["risky_choices"], 0.6)
