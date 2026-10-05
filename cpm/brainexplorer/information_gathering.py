import re

import numpy as np
import pandas as pd

from ._utils import load_data, quiet_empty_slices, require, session_time

__all__ = ["TreasureHunt"]

COLUMNS = ["userID", "date", "run", "outcome", "choseCurEv", "confidence", "draws", "median_diffRT", "confidenceRT"]


class TreasureHunt:
    """
    Compute descriptive statistics from the information-gathering task in BrainExplorer, *Treasure Hunt*.

    On each trial, participants draw samples of evidence until they decide to stop and choose an option, and then rate their confidence in their choice.

    Parameters
    ----------
    filepath : str, os.PathLike or pandas.DataFrame
        The data, as a DataFrame or the path to a CSV or Excel (``.xlsx``) file. The column names must follow the convention in Notes.

    Attributes
    ----------
    data : pandas.DataFrame
        The trials that pass the trial-level exclusion criteria.
    results : pandas.DataFrame
        The metrics of each participant, filled by :meth:`metrics`.
    cleanedresults : pandas.DataFrame
        The metrics of the participants that pass the participant-level exclusion criteria, filled by :meth:`clean_data`.
    deleted_participants : int
        The number of participants that :meth:`clean_data` excluded.
    codebook : dict
        The description of each column of the metrics.

    Examples
    --------
    >>> from cpm.brainexplorer.information_gathering import TreasureHunt
    >>> treasure_hunt = TreasureHunt("2025-02-20_TreasureHunt_Data.csv")
    >>> results = treasure_hunt.metrics()
    >>> cleaned = treasure_hunt.clean_data()
    >>> treasure_hunt.get_codebook()["mean_n_draws"]
    'Mean number of stimulus draws before making a decision'

    Notes
    -----
    The data must contain the following columns:

    - ``userID``: the unique identifier of the participant.
    - ``date``: the date and time of the trial.
    - ``run``: the attempt number of the participant.
    - ``outcome``: the points received on the trial.
    - ``confidence``: the confidence rating.
    - ``confidenceRT``: the response time of the confidence rating, in ms.
    - ``draws``: the number of draws on the trial.
    - ``choseCurEv``: whether the choice was in line with the current evidence (1) or not (0).
    - ``median_diffRT``: the median response time between the actions of the trial, in ms.

    If the data contain a column ``ev``, the evidence after each draw as a list of integers such as ``"[1 2 -1]"``, it is converted to lists of integers.

    Only the first attempt of each participant (``run`` equal to 1) is kept.
    """

    def __init__(self, filepath=None):
        data = load_data(filepath, "TreasureHunt")
        require(data, COLUMNS, "TreasureHunt")
        data = data[data["run"] == 1].copy()  # only keep the first attempt
        if "ev" in data:
            data["ev"] = data["ev"].apply(_parse_evidence)

        self.data = data
        self.results = pd.DataFrame()
        self.codebook = {
            "userID": "Unique identifier for each participant",
            "n_trials": "Number of trials completed by the participant",
            "date": "Date and time of the first trial",
            "day_of_week": "Day of the week of the first trial",
            "time": "Clock time of the first trial",
            "time_of_day": "Time of day of the first trial (night: 0-6h, morning: 6-12h, afternoon: 12-18h, evening: 18-24h)",
            "mean_points": "Mean points received across all trials",
            "accuracy": "Mean accuracy (proportion of choices in line with the current evidence)",
            "mean_confidence": "Mean confidence rating across all trials",
            "sd_confidence": "Standard deviation of confidence ratings across all trials",
            "mean_n_draws": "Mean number of stimulus draws before making a decision",
            "sd_n_draws": "Standard deviation of the number of stimulus draws before making a decision",
            "median_RT_between_actions": "Median reaction time between actions (ms)",
            "median_confidenceRT": "Median confidence reaction time across all trials (ms)",
            "unique_draws": "Number of unique values in number of draws",
        }

    def metrics(self):
        """
        Compute the metrics of each participant.

        Returns
        -------
        pandas.DataFrame
            The metrics, one row per participant, also stored in `results`. The columns are described in `codebook`.
        """
        rows = []
        with quiet_empty_slices():
            for user_id, user_data in self.data.groupby("userID"):
                row = {"userID": user_id, "n_trials": len(user_data)}
                row.update(session_time(user_data["date"].iloc[0]))
                row["mean_points"] = np.nanmean(user_data["outcome"])
                row["accuracy"] = np.nanmean(user_data["choseCurEv"])
                row["mean_confidence"] = np.nanmean(user_data["confidence"])
                row["sd_confidence"] = np.nanstd(user_data["confidence"])
                row["mean_n_draws"] = np.nanmean(user_data["draws"])
                row["sd_n_draws"] = np.nanstd(user_data["draws"])
                row["median_RT_between_actions"] = np.nanmedian(user_data["median_diffRT"])
                row["median_confidenceRT"] = np.nanmedian(user_data["confidenceRT"])
                row["unique_draws"] = user_data["draws"].nunique()
                rows.append(row)
        self.results = pd.DataFrame(rows)
        return self.results

    def clean_data(self):
        """
        Exclude participants with the participant-level exclusion criteria.

        Runs :meth:`metrics` first if it has not been run yet.

        Returns
        -------
        pandas.DataFrame
            The metrics of the participants that pass the criteria, also stored in `cleanedresults`.

        Notes
        -----
        A participant is excluded if

        - the mean number of draws is below 2 or above 23,
        - at least 20% of their choices were not in line with the current evidence (an accuracy of 0.8 or below),
        - the number of draws took fewer than 3 different values.
        """
        if self.results.empty:
            self.metrics()
        results = self.results
        n_before = results["userID"].nunique()

        keep = results["mean_n_draws"].between(2, 23)
        keep &= results["accuracy"] > 0.8
        keep &= results["unique_draws"] >= 3

        self.cleanedresults = results[keep].copy()
        self.deleted_participants = n_before - self.cleanedresults["userID"].nunique()
        return self.cleanedresults

    def get_codebook(self):
        """
        Return the codebook, which describes each column of the metrics.

        Returns
        -------
        dict
            The description of each column, keyed by column name.
        """
        return self.codebook


def _parse_evidence(value):
    """Turn evidence such as ``"[1 2 -1]"`` or ``"[1, 2, -1]"`` into a list of integers; other values are kept."""
    if isinstance(value, str):
        return [int(number) for number in re.findall(r"-?\d+", value)]
    return value
