import numpy as np
import pandas as pd

from cpm.brainexplorer.perceptual_decision_making import SpaceObserver


def _write_space_observer_csv(tmp_path, rows, filename="space_observer.csv"):
    data = pd.DataFrame(rows)
    filepath = tmp_path / filename
    data.to_csv(filepath, index=False)
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

    assert run1["n_trials"] == 4
    assert run1["time_of_day"] == "morning"
    assert np.isclose(run1["accuracy"], 0.5)
    assert np.isclose(run1["mean_RT"], 350)
    assert np.isclose(run1["median_RT"], 350)
    assert np.isclose(run1["median_RT_correct"], 300)
    assert np.isclose(run1["median_RT_incorrect"], 400)
    assert np.isclose(run1["mean_confidence"], 50)
    assert np.isclose(run1["median_confidence"], 50)
    assert np.isclose(run1["evidence_strength_mean"], 2.5)
    np.testing.assert_allclose(run1["ES_bins"], np.array([1.0, 2.0, 3.0, 4.0]))


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
    rows += _participant_rows(
        user_id="drop_acc",
        run=1,
        accuracies=[0, 0, 1, 0],
        rt_choices=[400, 500, 600, 700],
        confidences=[20, 30, 40, 50],
        confidence_rts=[500, 550, 600, 650],
        stimulus_intensities=[10, 12, 14, 16],
    )
    rows += _participant_rows(
        user_id="drop_flat_conf",
        run=1,
        accuracies=[1, 1, 1, 1],
        rt_choices=[400, 500, 600, 700],
        confidences=[50, 50, 50, 50],
        confidence_rts=[500, 550, 600, 650],
        stimulus_intensities=[10, 12, 14, 16],
    )
    rows += _participant_rows(
        user_id="drop_trials",
        run=1,
        accuracies=[1] * 81,
        rt_choices=[500] * 81,
        confidences=[60] * 81,
        confidence_rts=[500] * 81,
        stimulus_intensities=[12] * 81,
    )

    filepath = _write_space_observer_csv(tmp_path, rows)
    observer = SpaceObserver(str(filepath))
    observer.metrics()
    cleaned = observer.clean_data()

    remaining_users = set(cleaned["userID"].unique())
    assert remaining_users == {"keep"}
    assert observer.deleted_participants == 3
