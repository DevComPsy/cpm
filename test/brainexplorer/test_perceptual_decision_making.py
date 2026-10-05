import numpy as np
import pandas as pd
import pytest

from cpm.brainexplorer.perceptual_decision_making import SpaceObserver


def _write_space_observer_csv(tmp_path, rows):
    filepath = tmp_path / "space_observer.csv"
    pd.DataFrame(rows).to_csv(filepath, index=False)
    return filepath


def _participant_rows(
    user_id,
    run,
    accuracies,
    rt_choices,
    confidences,
    confidence_rts,
    stimulus_intensities,
    date="2025-02-20 09:00:00.000",
):
    rows = []
    for accuracy, rt_choice, confidence, confidence_rt, intensity in zip(
        accuracies,
        rt_choices,
        confidences,
        confidence_rts,
        stimulus_intensities,
    ):
        rows.append(
            {
                "userID": user_id,
                "run": run,
                "Date": date,
                "accuracy": accuracy,
                "RT_choice": rt_choice,
                "confidence": confidence,
                "confidenceRT": confidence_rt,
                "stimulus_intensity": intensity,
            }
        )
    return rows


def test_spaceobserver_init_filters_invalid_trials(tmp_path):
    rows = [
        {
            "userID": "u1",
            "run": 1,
            "Date": "2025-02-20 10:30:00.000",
            "accuracy": 1,
            "RT_choice": 200,
            "confidence": "50",
            "confidenceRT": 250,
            "stimulus_intensity": 2,
        },
        {
            "userID": "u1",
            "run": 1,
            "Date": "2025-02-20 10:30:00.000",
            "accuracy": 1,
            "RT_choice": 120,
            "confidence": "40",
            "confidenceRT": 250,
            "stimulus_intensity": 3,
        },
        {
            "userID": "u1",
            "run": 1,
            "Date": "2025-02-20 10:30:00.000",
            "accuracy": 1,
            "RT_choice": 250,
            "confidence": "40",
            "confidenceRT": 120,
            "stimulus_intensity": 3,
        },
        {
            "userID": "u1",
            "run": 1,
            "Date": "2025-02-20 10:30:00.000",
            "accuracy": 1,
            "RT_choice": 250,
            "confidence": "",
            "confidenceRT": 250,
            "stimulus_intensity": 3,
        },
    ]

    filepath = _write_space_observer_csv(tmp_path, rows)
    observer = SpaceObserver(str(filepath))

    assert len(observer.data_raw) == 1
    assert observer.data_raw.iloc[0]["RT_choice"] == 200
    assert observer.data_raw.iloc[0]["confidenceRT"] == 250
    assert observer.data_raw.iloc[0]["confidence"] == 50
    assert pd.api.types.is_numeric_dtype(observer.data_raw["confidence"])


def test_spaceobserver_time_of_day(tmp_path):
    base_row = {
        "userID": "u1",
        "run": 1,
        "accuracy": 1,
        "RT_choice": 300,
        "confidence": "50",
        "confidenceRT": 300,
        "stimulus_intensity": 5,
    }
    cases = [
        ("2025-02-20 02:00:00.000", "night"),
        ("2025-02-20 09:00:00.000", "morning"),
        ("2025-02-20 14:00:00.000", "afternoon"),
        ("2025-02-20 20:00:00.000", "evening"),
    ]
    for date_str, expected_tod in cases:
        rows = [{**base_row, "Date": date_str}]
        filepath = tmp_path / f"tod_{expected_tod}.csv"
        pd.DataFrame(rows).to_csv(filepath, index=False)
        observer = SpaceObserver(str(filepath))
        results = observer.metrics()
        assert results.iloc[0]["time_of_day"] == expected_tod, (
            f"Expected time_of_day={expected_tod!r} for {date_str}, "
            f"got {results.iloc[0]['time_of_day']!r}"
        )


def test_spaceobserver_metrics_computes_expected_values(tmp_path):
    rows = _participant_rows(
        user_id="u1",
        run=1,
        accuracies=[1, 0, 1, 0],
        rt_choices=[200, 300, 400, 500],
        confidences=[20, 40, 60, 80],
        confidence_rts=[250, 260, 270, 280],
        stimulus_intensities=[1, 2, 3, 4],
        date="2025-02-20 10:30:00.000",
    )
    rows += _participant_rows(
        user_id="u1",
        run=2,
        accuracies=[1, 1],
        rt_choices=[600, 700],
        confidences=[90, 95],
        confidence_rts=[300, 310],
        stimulus_intensities=[10, 12],
        date="2025-02-20 18:30:00.000",
    )

    filepath = _write_space_observer_csv(tmp_path, rows)
    observer = SpaceObserver(str(filepath))
    results = observer.metrics().sort_values(["userID", "run"]).reset_index(drop=True)

    assert len(results) == 2
    run1 = results[results["run"] == 1].iloc[0]
    run2 = results[results["run"] == 2].iloc[0]

    # run 1 basic metrics
    assert run1["n_trials"] == 4
    assert run1["time_of_day"] == "morning"
    assert np.isclose(run1["accuracy"], 0.5)

    # RT metrics
    assert np.isclose(run1["mean_RT"], 350)
    assert np.isclose(run1["median_RT"], 350)
    assert np.isclose(run1["median_RT_correct"], 300)   # median([200, 400])
    assert np.isclose(run1["median_RT_incorrect"], 400)  # median([300, 500])
    assert np.isclose(run1["diff_median_RT_correct_incorrect"], -100)

    # confidence metrics
    assert np.isclose(run1["mean_confidence"], 50)
    assert np.isclose(run1["median_confidence"], 50)
    assert np.isclose(run1["mean_confidence_correct"], 40)    # mean([20, 60])
    assert np.isclose(run1["mean_confidence_incorrect"], 60)  # mean([40, 80])
    assert np.isclose(run1["diff_median_conf_correct_incorrect"], -20)
    assert np.isclose(run1["sd_confidence"], np.std([20, 40, 60, 80]))

    # confidence percentiles
    assert np.isclose(run1["confidence_10"], np.percentile([20, 40, 60, 80], 10))
    assert np.isclose(run1["confidence_25"], np.percentile([20, 40, 60, 80], 25))
    assert np.isclose(run1["confidence_75"], np.percentile([20, 40, 60, 80], 75))
    assert np.isclose(run1["confidence_90"], np.percentile([20, 40, 60, 80], 90))

    # confidenceRT metrics
    assert np.isclose(run1["median_confidenceRT"], 265)        # median([250,260,270,280])
    assert np.isclose(run1["median_confidenceRT_correct"], 260)   # median([250, 270])
    assert np.isclose(run1["median_confidenceRT_incorrect"], 270)  # median([260, 280])
    assert np.isclose(run1["diff_median_confidenceRT_correct_incorrect"], -10)

    # evidence strength metrics
    assert np.isclose(run1["evidence_strength_mean"], 2.5)
    assert np.isclose(run1["evidence_strength_correct_mean"], 2.0)   # mean([1, 3])
    assert np.isclose(run1["evidence_strength_incorrect_mean"], 3.0)  # mean([2, 4])
    assert np.isclose(run1["diff_evidence_strength_correct_incorrect"], -1.0)
    assert np.isclose(run1["median_ES"], 2.5)
    bins = [run1[f"ES_bin_{i}"] for i in range(1, 5)]
    np.testing.assert_allclose(bins, [1.0, 2.0, 3.0, 4.0])

    # run 2 has time_of_day == evening (18:30)
    assert run2["time_of_day"] == "evening"
    assert run2["n_trials"] == 2
    assert np.isclose(run2["accuracy"], 1.0)


def test_spaceobserver_clean_data_excludes_invalid_participants(tmp_path):
    rows = []
    rows += _participant_rows(
        user_id="keep",
        run=1,
        accuracies=[1, 1, 0, 1],
        rt_choices=[400, 500, 600, 700],
        confidences=[20, 30, 40, 50],
        confidence_rts=[500, 550, 600, 650],
        stimulus_intensities=[10, 12, 14, 16],
    )
    # accuracy < 0.5
    rows += _participant_rows(
        user_id="drop_acc",
        run=1,
        accuracies=[0, 0, 1, 0],
        rt_choices=[400, 500, 600, 700],
        confidences=[20, 30, 40, 50],
        confidence_rts=[500, 550, 600, 650],
        stimulus_intensities=[10, 12, 14, 16],
    )
    # flat confidence (10th==25th and 75th==90th)
    rows += _participant_rows(
        user_id="drop_flat_conf",
        run=1,
        accuracies=[1, 1, 1, 1],
        rt_choices=[400, 500, 600, 700],
        confidences=[50, 50, 50, 50],
        confidence_rts=[500, 550, 600, 650],
        stimulus_intensities=[10, 12, 14, 16],
    )
    # more than 80 trials
    rows += _participant_rows(
        user_id="drop_trials",
        run=1,
        accuracies=[1] * 81,
        rt_choices=[500] * 81,
        confidences=[60] * 81,
        confidence_rts=[500] * 81,
        stimulus_intensities=[12] * 81,
    )
    # median RT > 3000 ms
    rows += _participant_rows(
        user_id="drop_slow_rt",
        run=1,
        accuracies=[1, 1, 0, 1],
        rt_choices=[3200, 3100, 3300, 3400],
        confidences=[30, 40, 50, 60],
        confidence_rts=[500, 550, 600, 650],
        stimulus_intensities=[10, 12, 14, 16],
    )
    # median confidenceRT > 3000 ms
    rows += _participant_rows(
        user_id="drop_slow_conf_rt",
        run=1,
        accuracies=[1, 1, 0, 1],
        rt_choices=[400, 500, 600, 700],
        confidences=[30, 40, 50, 60],
        confidence_rts=[3100, 3200, 3300, 3400],
        stimulus_intensities=[10, 12, 14, 16],
    )
    # median evidence strength > 25
    rows += _participant_rows(
        user_id="drop_high_es",
        run=1,
        accuracies=[1, 1, 0, 1],
        rt_choices=[400, 500, 600, 700],
        confidences=[30, 40, 50, 60],
        confidence_rts=[500, 550, 600, 650],
        stimulus_intensities=[26, 28, 30, 32],
    )
    # median confidence < 3
    rows += _participant_rows(
        user_id="drop_low_conf",
        run=1,
        accuracies=[1, 1, 0, 1],
        rt_choices=[400, 500, 600, 700],
        confidences=[1, 2, 1, 2],
        confidence_rts=[500, 550, 600, 650],
        stimulus_intensities=[10, 12, 14, 16],
    )
    # median confidence > 97
    rows += _participant_rows(
        user_id="drop_high_conf",
        run=1,
        accuracies=[1, 1, 0, 1],
        rt_choices=[400, 500, 600, 700],
        confidences=[97, 99, 97, 99],
        confidence_rts=[500, 550, 600, 650],
        stimulus_intensities=[10, 12, 14, 16],
    )

    filepath = _write_space_observer_csv(tmp_path, rows)
    observer = SpaceObserver(str(filepath))
    observer.metrics()
    cleaned = observer.clean_data()

    remaining_users = set(cleaned["userID"].unique())
    assert remaining_users == {"keep"}
    assert observer.deleted_participants == 8


def _keep_rows(user_id="keep", run=1, date="2025-02-20 09:00:00.000"):
    return _participant_rows(
        user_id=user_id,
        run=run,
        accuracies=[1, 1, 0, 1],
        rt_choices=[400, 500, 600, 700],
        confidences=[20, 30, 40, 50],
        confidence_rts=[500, 550, 600, 650],
        stimulus_intensities=[10, 12, 14, 16],
        date=date,
    )


def test_spaceobserver_accepts_dataframe_and_path(tmp_path):
    rows = _keep_rows()
    filepath = _write_space_observer_csv(tmp_path, rows)
    from_path = SpaceObserver(filepath).metrics()
    from_frame = SpaceObserver(pd.DataFrame(rows)).metrics()
    pd.testing.assert_frame_equal(from_path, from_frame)


def test_spaceobserver_does_not_change_the_input_dataframe():
    data = pd.DataFrame(_keep_rows())
    original = data.copy()
    SpaceObserver(data).metrics()
    pd.testing.assert_frame_equal(data, original)


def test_spaceobserver_requires_data_and_columns():
    with pytest.raises(ValueError, match="needs data"):
        SpaceObserver()
    data = pd.DataFrame(_keep_rows()).drop(columns=["confidenceRT"])
    with pytest.raises(KeyError, match="confidenceRT"):
        SpaceObserver(data)


def test_spaceobserver_metrics_can_be_run_twice():
    observer = SpaceObserver(pd.DataFrame(_keep_rows()))
    first = observer.metrics().copy()
    second = observer.metrics()
    pd.testing.assert_frame_equal(first, second)
    assert len(second) == 1


def test_spaceobserver_dates_without_fractional_seconds():
    observer = SpaceObserver(pd.DataFrame(_keep_rows(date="2025-02-22 23:15:00")))
    result = observer.metrics().iloc[0]
    assert result["day_of_week"] == "Saturday"
    assert result["time_of_day"] == "evening"
    assert str(result["time"]) == "23:15:00"


def test_spaceobserver_all_correct_session_is_nan_for_incorrect_trials(recwarn):
    rows = _participant_rows(
        user_id="u1",
        run=1,
        accuracies=[1, 1, 1],
        rt_choices=[300, 400, 500],
        confidences=[50, 60, 70],
        confidence_rts=[300, 300, 300],
        stimulus_intensities=[5, 6, 7],
    )
    result = SpaceObserver(pd.DataFrame(rows)).metrics().iloc[0]
    assert np.isnan(result["median_RT_incorrect"])
    assert np.isnan(result["mean_confidence_incorrect"])
    assert np.isnan(result["diff_median_RT_correct_incorrect"])
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_spaceobserver_clean_data_runs_metrics_first():
    observer = SpaceObserver(pd.DataFrame(_keep_rows()))
    cleaned = observer.clean_data()
    assert list(cleaned["userID"]) == ["keep"]
    assert observer.deleted_participants == 0


def test_spaceobserver_trial_limit_counts_all_sessions():
    # 4 sessions of 21 trials: no session has more than 80 trials, but the participant has 84
    rows = []
    for run in range(1, 5):
        rows += _participant_rows(
            user_id="many_sessions",
            run=run,
            accuracies=[1, 0, 1] * 7,
            rt_choices=[400, 500, 600] * 7,
            confidences=[20, 50, 80] * 7,
            confidence_rts=[500, 550, 600] * 7,
            stimulus_intensities=[10, 12, 14] * 7,
        )
    rows += _keep_rows()
    observer = SpaceObserver(pd.DataFrame(rows))
    cleaned = observer.clean_data()
    assert set(cleaned["userID"]) == {"keep"}
    assert observer.deleted_participants == 1


def test_spaceobserver_codebook_describes_every_metric():
    observer = SpaceObserver(pd.DataFrame(_keep_rows()))
    results = observer.metrics()
    assert set(results.columns) == set(observer.get_codebook())
