import datetime

import numpy as np
import pandas as pd
import pytest

from cpm.brainexplorer._utils import load_data, proportion, require, session_time, time_of_day


@pytest.mark.parametrize(
    "hour, expected",
    [(0, "night"), (5, "night"), (6, "morning"), (11, "morning"), (12, "afternoon"), (17, "afternoon"), (18, "evening"), (23, "evening")],
)
def test_time_of_day(hour, expected):
    assert time_of_day(hour) == expected


@pytest.mark.parametrize(
    "date",
    ["2025-02-20 19:05:30.250", "2025-02-20 19:05:30", "2025-02-20T19:05:30.250", pd.Timestamp("2025-02-20 19:05:30.250")],
)
def test_session_time(date):
    out = session_time(date)
    assert out["date"].floor("s") == pd.Timestamp("2025-02-20 19:05:30")
    assert out["day_of_week"] == "Thursday"
    assert out["time"].replace(microsecond=0) == datetime.time(19, 5, 30)
    assert out["time_of_day"] == "evening"


def test_load_data_copies_dataframes():
    data = pd.DataFrame({"a": [1, 2]})
    loaded = load_data(data, "Task")
    loaded.loc[0, "a"] = 10
    assert data.loc[0, "a"] == 1


def test_load_data_reads_csv(tmp_path):
    filepath = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(filepath, index=False)
    pd.testing.assert_frame_equal(load_data(filepath, "Task"), pd.DataFrame({"a": [1, 2]}))
    pd.testing.assert_frame_equal(load_data(str(filepath), "Task"), pd.DataFrame({"a": [1, 2]}))


def test_load_data_needs_a_source():
    with pytest.raises(ValueError, match="Task needs data"):
        load_data(None, "Task")


def test_require_names_every_missing_column():
    with pytest.raises(KeyError, match="'b', 'c'"):
        require(pd.DataFrame({"a": [1]}), ["a", "b", "c"], "Task")


def test_proportion():
    assert proportion([True, False, True, True], [True, True, True, False]) == pytest.approx(2 / 3)
    assert np.isnan(proportion([True, False], [False, False]))
