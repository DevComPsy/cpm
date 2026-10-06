"""Helpers shared by the BrainExplorer task classes."""

import os
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd


def load_data(source, name, **kwargs):
    """
    Read the data of a task from a CSV file, or copy a DataFrame.

    `source` is a pandas DataFrame, or the path to a CSV file. Keyword arguments
    are passed on to :func:`pandas.read_csv`.
    """
    if source is None:
        raise ValueError(f"{name} needs data: a pandas DataFrame or the path to a CSV file.")
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(os.fspath(source), header=0, **kwargs)


def require(data, columns, name):
    """Raise a `KeyError` that names every required column missing from `data`."""
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(f"{name} needs the columns {missing}, which are missing from the data.")


def time_of_day(hour):
    """Name the time of day of an hour: night (0-6h), morning (6-12h), afternoon (12-18h) or evening (18-24h)."""
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def session_time(date):
    """The date, day of the week, clock time and time of day of the first trial of a session."""
    date = pd.Timestamp(pd.to_datetime(date, format="ISO8601")) if isinstance(date, str) else pd.Timestamp(date)
    return {
        "date": date,
        "day_of_week": date.day_name(),
        "time": date.time(),
        "time_of_day": time_of_day(date.hour),
    }


@contextmanager
def quiet_empty_slices():
    """Silence the warnings numpy raises for the mean, median or SD of an empty selection, which are NaN by design."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
        warnings.filterwarnings("ignore", message="invalid value encountered")
        yield


def proportion(condition, mask):
    """The proportion of trials in `mask` for which `condition` is true, or NaN if `mask` selects no trials."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.nan
    return float(np.mean(np.asarray(condition, dtype=bool)[mask]))
