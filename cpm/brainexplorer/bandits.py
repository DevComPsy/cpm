import numpy as np
import pandas as pd

from ._utils import load_data, quiet_empty_slices, require, session_time

__all__ = ["MilkyWay"]

COLUMNS = ["userID", "trial_type", "run", "date", "correct", "outchosen", "obt_min_forg", "rep", "WSLS_v1", "WSLS_v2"]

## the two games, by their trial_type in the data: (name, suffix of the metrics)
GAMES = {"reward": ("Milky Way", "MW"), "punish": ("Pirate Market", "PM")}

## the metrics that are compared between the games: (name of the difference, metric without its suffix)
DIFFERENCES = [
    ("accuracy_diff", "accuracy"),
    ("outcome_diff", "mean_outcome"),
    ("reward_diff_obt_forg_diff", "reward_diff_obt_forg"),
    ("prop_same_choice_diff", "prop_same_choice"),
    ("prop_WSLS_1_diff", "prop_WSLS_1"),
    ("prop_WSLS_2_diff", "prop_WSLS_2"),
]


class MilkyWay:
    """
    Compute descriptive statistics from the two-armed bandit tasks in BrainExplorer, *Milky Way* (reward) and *Pirate Market* (punishment).

    The metrics of the two games are computed separately, and :meth:`difference_metrics` compares them within participants.

    Parameters
    ----------
    filepath : str, os.PathLike or pandas.DataFrame
        The data, as a DataFrame or the path to a CSV file. The column names must follow the convention in Notes.

    Attributes
    ----------
    MW_data, PM_data : pandas.DataFrame
        The Milky Way and Pirate Market trials that pass the trial-level exclusion criteria.
    results_MW, results_PM : pandas.DataFrame
        The metrics of each participant in each game, filled by :meth:`metrics`.
    results_diff : pandas.DataFrame
        The differences between the games, filled by :meth:`difference_metrics`.
    cleanedresults_MW, cleanedresults_PM, cleanedresults_diff : pandas.DataFrame
        The results of the participants that pass the participant-level exclusion criteria, filled by :meth:`clean_data`.
    deleted_participants_MW, deleted_participants_PM, deleted_participants_diff : int
        The number of participants that :meth:`clean_data` excluded from each table.
    codebook : dict
        The description of each column of the metrics.

    Examples
    --------
    >>> from cpm.brainexplorer.bandits import MilkyWay
    >>> milky_way = MilkyWay("2025-02-20_MilkyWay_Data.csv")
    >>> results_MW, results_PM = milky_way.metrics()
    >>> differences = milky_way.difference_metrics()
    >>> cleaned_MW, cleaned_PM, cleaned_diff = milky_way.clean_data()

    Notes
    -----
    The data must contain the following columns:

    - ``userID``: the unique identifier of the participant.
    - ``trial_type``: the game, ``"reward"`` for Milky Way or ``"punish"`` for Pirate Market.
    - ``run``: the attempt number of the participant.
    - ``date``: the date and time of the trial.
    - ``correct``: whether the choice was correct (1) or incorrect (0).
    - ``outchosen``: the outcome of the chosen option.
    - ``obt_min_forg``: the obtained minus the foregone outcome.
    - ``rep``: whether the choice repeated the previous one (1) or not (0).
    - ``WSLS_v1``, ``WSLS_v2``: whether the choice was a win-stay/lose-shift choice, in two versions of the definition.

    Only the first attempt of each participant in each game is kept, which is not necessarily ``run`` 1 for Pirate Market.

    Response times are not analysed. They would have to be corrected for the average time between the previous choice and the presentation of the new stimuli, 4200 ms, which can make them negative.
    """

    def __init__(self, filepath=None):
        data = load_data(filepath, "MilkyWay")
        require(data, COLUMNS, "MilkyWay")
        self.data = data
        self.MW_data = _first_attempt(data[data["trial_type"] == "reward"])
        self.PM_data = _first_attempt(data[data["trial_type"] == "punish"])

        self.results_MW = pd.DataFrame()
        self.results_PM = pd.DataFrame()
        self.results_diff = pd.DataFrame()

        self.codebook = {
            "userID": "Unique identifier for each participant",
            "n_trials": "Number of trials completed by the participant",
            "trial_type": "Type of trial (Milky Way or Pirate Market)",
            "date": "Date and time of the first trial",
            "day_of_week": "Day of the week of the first trial",
            "time": "Clock time of the first trial",
            "time_of_day": "Time of day of the first trial (night: 0-6h, morning: 6-12h, afternoon: 12-18h, evening: 18-24h)",
        }
        for name, suffix in GAMES.values():
            self.codebook.update(
                {
                    f"accuracy_{suffix}": f"Mean accuracy for {name} trials",
                    f"mean_outcome_{suffix}": f"Mean outcome for {name} trials",
                    f"reward_diff_obt_forg_{suffix}": f"Mean difference between obtained and foregone reward for {name} trials",
                    f"prop_same_choice_{suffix}": f"Proportion of same choice in {name} trials",
                    f"prop_WSLS_1_{suffix}": f"Proportion of win-stay/lose-shift choices (version 1) for {name} trials",
                    f"prop_WSLS_2_{suffix}": f"Proportion of win-stay/lose-shift choices (version 2) for {name} trials",
                }
            )
        self.codebook.update(
            {
                "accuracy_diff": "Difference in accuracy: Milky Way minus Pirate Market",
                "outcome_diff": "Difference in mean outcome: Milky Way minus Pirate Market",
                "reward_diff_obt_forg_diff": "Difference in the mean obtained minus foregone reward: Milky Way minus Pirate Market",
                "prop_same_choice_diff": "Difference in the proportion of same choices: Milky Way minus Pirate Market",
                "prop_WSLS_1_diff": "Difference in the proportion of win-stay/lose-shift choices (version 1): Milky Way minus Pirate Market",
                "prop_WSLS_2_diff": "Difference in the proportion of win-stay/lose-shift choices (version 2): Milky Way minus Pirate Market",
            }
        )

    def metrics(self):
        """
        Compute the metrics of each participant in each game.

        Returns
        -------
        tuple of pandas.DataFrame
            The metrics of Milky Way and of Pirate Market, one row per participant, also stored in `results_MW` and `results_PM`. The columns are described in `codebook`.
        """
        self.results_MW = _game_metrics(self.MW_data, *GAMES["reward"])
        self.results_PM = _game_metrics(self.PM_data, *GAMES["punish"])
        return self.results_MW, self.results_PM

    def difference_metrics(self):
        """
        Compute the difference between the metrics of Milky Way and Pirate Market, for each participant who played both.

        Runs :meth:`metrics` first if it has not been run yet.

        Returns
        -------
        pandas.DataFrame
            The differences, Milky Way minus Pirate Market, one row per participant, also stored in `results_diff`. The columns are described in `codebook`.
        """
        if self.results_MW.empty and self.results_PM.empty:
            self.metrics()
        columns = ["userID"] + [name for name, _ in DIFFERENCES]
        if self.results_MW.empty or self.results_PM.empty:
            self.results_diff = pd.DataFrame(columns=columns)
            return self.results_diff
        both = self.results_MW.merge(self.results_PM, on="userID", sort=False)
        for name, metric in DIFFERENCES:
            both[name] = both[f"{metric}_MW"] - both[f"{metric}_PM"]
        self.results_diff = both[columns].reset_index(drop=True)
        return self.results_diff

    def clean_data(self):
        """
        Exclude participants with the participant-level exclusion criteria.

        Runs :meth:`metrics` and :meth:`difference_metrics` first if they have not been run yet.

        Returns
        -------
        tuple of pandas.DataFrame
            The metrics of Milky Way, of Pirate Market, and their differences, for the participants that pass the criteria, also stored in `cleanedresults_MW`, `cleanedresults_PM` and `cleanedresults_diff`.

        Notes
        -----
        A participant is excluded from the metrics of a game if

        - they made the same choice on at least 95% of the trials,
        - their accuracy is missing,
        - they played more than 72 trials of the game (due to a technical error).

        The differences are kept only for participants who pass the criteria in both games.
        """
        if self.results_MW.empty and self.results_PM.empty:
            self.metrics()
        if self.results_diff.empty:
            self.difference_metrics()

        self.cleanedresults_MW = _clean_game(self.results_MW, self.MW_data, "MW")
        self.cleanedresults_PM = _clean_game(self.results_PM, self.PM_data, "PM")
        self.deleted_participants_MW = _n_users(self.results_MW) - _n_users(self.cleanedresults_MW)
        self.deleted_participants_PM = _n_users(self.results_PM) - _n_users(self.cleanedresults_PM)

        both = set(_users(self.cleanedresults_MW)) & set(_users(self.cleanedresults_PM))
        self.cleanedresults_diff = self.results_diff[self.results_diff["userID"].isin(both)].copy()
        self.deleted_participants_diff = _n_users(self.results_diff) - _n_users(self.cleanedresults_diff)

        return self.cleanedresults_MW, self.cleanedresults_PM, self.cleanedresults_diff

    def get_codebook(self):
        """
        Return the codebook, which describes each column of the metrics.

        Returns
        -------
        dict
            The description of each column, keyed by column name.
        """
        return self.codebook


def _first_attempt(data):
    """The trials of the first attempt of each participant, which is their lowest `run`."""
    first = data.groupby("userID")["run"].transform("min")
    return data[data["run"] == first].reset_index(drop=True)


def _game_metrics(data, name, suffix):
    """The metrics of each participant in one game."""
    rows = []
    with quiet_empty_slices():
        for user_id, user_data in data.groupby("userID"):
            row = {"userID": user_id, "n_trials": len(user_data), "trial_type": name}
            row.update(session_time(user_data["date"].iloc[0]))
            row[f"accuracy_{suffix}"] = np.nanmean(user_data["correct"])
            row[f"mean_outcome_{suffix}"] = np.nanmean(user_data["outchosen"])
            row[f"reward_diff_obt_forg_{suffix}"] = np.nanmean(user_data["obt_min_forg"])
            row[f"prop_same_choice_{suffix}"] = np.nanmean(user_data["rep"])
            row[f"prop_WSLS_1_{suffix}"] = np.nanmean(user_data["WSLS_v1"])
            row[f"prop_WSLS_2_{suffix}"] = np.nanmean(user_data["WSLS_v2"])
            rows.append(row)
    return pd.DataFrame(rows)


def _clean_game(results, data, suffix):
    """The metrics of one game, for the participants that pass the exclusion criteria."""
    if results.empty:
        return results.copy()
    trials = data.groupby("userID").size()
    keep = results["userID"].isin(trials.index[trials <= 72])
    keep &= results[f"prop_same_choice_{suffix}"] < 0.95
    keep &= results[f"accuracy_{suffix}"].notna()
    return results[keep].copy()


def _users(results):
    return results["userID"].unique() if "userID" in results else []


def _n_users(results):
    return len(_users(results))
