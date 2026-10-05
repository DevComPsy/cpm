import numpy as np
import pandas as pd

from ._utils import load_data, proportion, quiet_empty_slices, require, session_time

__all__ = ["Scavenger"]

COLUMNS = [
    "userID",
    "run",
    "date",
    "RT",
    "chosen",
    "ambiguousTrial",
    "abovechance",
    "EV_diff_chosen",
    "safe_EV",
    "outcome",
    "nbcorrect_gain",
    "resp_rep",
    "resp_rep_sr",
]


class Scavenger:
    """
    Compute descriptive statistics from the risky decision-making task in BrainExplorer, *Scavenger*.

    On each trial, participants choose between two options: the safe option returns its outcome with 100% probability, while the risky option returns one of two outcomes with 50% probability each.
    In loss trials, the expected value of the safe option is negative.
    In ambiguous trials, the probabilities of the outcomes of the risky option are only partly shown.

    Parameters
    ----------
    filepath : str, os.PathLike or pandas.DataFrame
        The data, as a DataFrame or the path to a CSV or Excel (``.xlsx``) file. The column names must follow the convention in Notes.

    Attributes
    ----------
    data : pandas.DataFrame
        The trials that pass the trial-level exclusion criteria, with an added column ``risky``, 1 for risky and 0 for safe choices.
    results : pandas.DataFrame
        The metrics of each participant, filled by :meth:`metrics`.
    cleanedresults : pandas.DataFrame
        The metrics of the participants that pass the participant-level exclusion criteria, filled by :meth:`clean_data`.
    deleted_participants : int
        The number of participants that :meth:`clean_data` excluded.
    codebook : dict
        The description of each column of the metrics.

    Raises
    ------
    ValueError
        If ``chosen`` contains values other than 1 (safe) and 2 (risky).

    Examples
    --------
    >>> from cpm.brainexplorer.risky_decision_making import Scavenger
    >>> scavenger = Scavenger("2025-01-10_Scavenger.xlsx")
    >>> results = scavenger.metrics()
    >>> cleaned = scavenger.clean_data()
    >>> scavenger.get_codebook()["risky_choices"]
    'Proportion of risky choices overall'

    Notes
    -----
    The data must contain the following columns:

    - ``userID``: the unique identifier of the participant.
    - ``run``: the attempt number of the participant.
    - ``date``: the date and time of the trial.
    - ``RT``: the response time, in ms.
    - ``chosen``: the chosen option, 1 for safe and 2 for risky.
    - ``ambiguousTrial``: whether the trial was ambiguous (1) or not (0).
    - ``abovechance``: whether the choice was above chance level (1) or not (0).
    - ``EV_diff_chosen``: the expected value of the chosen minus the unchosen option. A choice is rational if it is 0 or more.
    - ``safe_EV``: the expected value of the safe option; trials where it is negative are loss trials, the others win trials.
    - ``outcome``: the points received on the trial.
    - ``nbcorrect_gain``: whether the choice in a no-brainer gain (catch) trial was correct (1) or not (0).
    - ``resp_rep``: whether the left/right response repeated the previous one (1) or not (0).
    - ``resp_rep_sr``: whether the safe/risky choice repeated the previous one (1) or not (0).
    - ``config_id``: the version of the game. Only needed by :meth:`clean_data`.

    Trials are excluded if the response time is below 150 ms or above 10000 ms, and only the first attempt of each participant (``run`` equal to 1) is kept.

    References
    ----------
    Habicht, J., Dubois, M., Michely, J., & Hauser, T. U. (2022). Do propranolol and amisulpride modulate confidence in risk-taking? *Wellcome Open Research*, 7, 23.
    """

    def __init__(self, filepath=None):
        data = load_data(filepath, "Scavenger")
        require(data, COLUMNS, "Scavenger")
        data = data[data["run"] == 1]  # only keep the first attempt
        data = data[data["RT"].between(150, 10000)].copy()

        codes = set(data["chosen"].dropna().unique())
        if not codes <= {1, 2}:
            raise ValueError(
                f"Scavenger expects 'chosen' to be 1 (safe) or 2 (risky), but the data contain {sorted(codes)}."
            )
        data["risky"] = data["chosen"] - 1

        self.data = data
        self.results = pd.DataFrame()
        self.codebook = {
            # --- Participant info ---
            "userID": "Unique identifier for each participant",
            "n_trials": "Number of trials completed by the participant",
            "date": "Date and time of the first trial",
            "day_of_week": "Day of the week the session started",
            "time": "Clock time of the first trial",
            "time_of_day": "Time of day of the session (night: 0-6h, morning: 6-12h, afternoon: 12-18h, evening: 18-24h)",
            # --- Reaction time ---
            "mean_RT": "Mean reaction time across all trials (ms)",
            "median_RT": "Median reaction time across all trials (ms)",
            "median_RT_safe": "Median reaction time for safe choices (ms)",
            "median_RT_risky": "Median reaction time for risky choices (ms)",
            "median_RT_win": "Median reaction time for win trials (ms)",
            "median_RT_loss": "Median reaction time for loss trials (ms)",
            "median_RT_ambg": "Median reaction time for ambiguous trials (ms)",
            "median_RT_NoAmbg": "Median reaction time for non-ambiguous trials (ms)",
            "median_RT_win_ambg": "Median reaction time for win trials that were ambiguous (ms)",
            "median_RT_win_NoAmbg": "Median reaction time for win trials that were non-ambiguous (ms)",
            "median_RT_loss_ambg": "Median reaction time for loss trials that were ambiguous (ms)",
            "median_RT_loss_NoAmbg": "Median reaction time for loss trials that were non-ambiguous (ms)",
            "diff_RT_win_amg_win_noAmbg": "Difference in median RT: win-ambiguous minus win-non-ambiguous (ms)",
            "diff_RT_loss_amg_loss_noAmbg": "Difference in median RT: loss-ambiguous minus loss-non-ambiguous (ms)",
            "diff_RT_win_loss_ambg": "Difference in median RT: win minus loss in ambiguous trials (ms)",
            "diff_RT_win_loss_noAmbg": "Difference in median RT: win minus loss in non-ambiguous trials (ms)",
            "diff_RT_win_ambg_loss_noAmbg": "Difference in median RT: win-ambiguous minus loss-non-ambiguous (ms)",
            "diff_RT_loss_ambg_win_noAmbg": "Difference in median RT: loss-ambiguous minus win-non-ambiguous (ms)",
            # --- Points / outcomes ---
            "total_rewards": "Total points received across all trials",
            "mean_points": "Mean points received across all trials",
            "mean_points_safe": "Mean points received when choosing safe",
            "mean_points_risky": "Mean points received when choosing risky",
            "mean_points_ambg": "Mean points received in ambiguous trials",
            "mean_points_NoAmbg": "Mean points received in non-ambiguous trials",
            "mean_points_win_ambg": "Mean points received in win trials that were ambiguous",
            "mean_points_win_NoAmbg": "Mean points received in win trials that were non-ambiguous",
            "mean_points_loss_ambg": "Mean points received in loss trials that were ambiguous",
            "mean_points_loss_NoAmbg": "Mean points received in loss trials that were non-ambiguous",
            "diff_points_win_amg_win_noAmbg": "Difference in mean points: win-ambiguous minus win-non-ambiguous",
            "diff_points_loss_amg_loss_noAmbg": "Difference in mean points: loss-ambiguous minus loss-non-ambiguous",
            "diff_points_win_loss_ambg": "Difference in mean points: win minus loss in ambiguous trials",
            "diff_points_win_loss_noAmbg": "Difference in mean points: win minus loss in non-ambiguous trials",
            "diff_points_win_ambg_loss_noAmbg": "Difference in mean points: win-ambiguous minus loss-non-ambiguous",
            "diff_points_loss_ambg_win_noAmbg": "Difference in mean points: loss-ambiguous minus win-non-ambiguous",
            # --- Choice stickiness ---
            "choice_stickiness": "Number of consecutive trial pairs where the same option (safe or risky) was chosen",
            # --- Risky choice rates ---
            "risky_choices": "Proportion of risky choices overall",
            "win_risk": "Proportion of risky choices in win trials",
            "loss_risk": "Proportion of risky choices in loss trials",
            "diff_loss_win_risk": "Difference in risky choice rate: loss minus win trials",
            "risk_Ambg": "Proportion of risky choices in ambiguous trials",
            "risk_NoAmbg": "Proportion of risky choices in non-ambiguous trials",
            "diff_Ambg_NoAmbg_risk": "Difference in risky choice rate: ambiguous minus non-ambiguous trials",
            "win_risk_abg": "Proportion of risky choices in win-ambiguous trials",
            "win_risk_NoAbg": "Proportion of risky choices in win-non-ambiguous trials",
            "loss_risk_abg": "Proportion of risky choices in loss-ambiguous trials",
            "loss_risk_NoAbg": "Proportion of risky choices in loss-non-ambiguous trials",
            "diff_Ambg_NoAmbg_win_risk": "Difference in risky win rate: ambiguous minus non-ambiguous",
            "diff_Ambg_NoAmbg_loss_risk": "Difference in risky loss rate: ambiguous minus non-ambiguous",
            "diff_Ambg_win_loss_risk": "Difference in risky choice rate: win minus loss in ambiguous trials",
            "diff_NoAmbg_win_loss_risk": "Difference in risky choice rate: win minus loss in non-ambiguous trials",
            "diff_amg_win_nonAmg_loss_risk": "Difference in risky choice rate: win-ambiguous minus loss-non-ambiguous",
            "diff_amg_loss_nonAmg_win_risk": "Difference in risky choice rate: loss-ambiguous minus win-non-ambiguous",
            # --- Rational choices (overall, win/loss, ambiguity) ---
            "rational_all": "Proportion of rational choices across all trials",
            "rational_win": "Proportion of rational choices in win trials",
            "rational_loss": "Proportion of rational choices in loss trials",
            "diff_loss_win_rational": "Difference in rational choice rate: loss minus win trials",
            "rational_abg": "Proportion of rational choices in ambiguous trials",
            "rational_NoAbg": "Proportion of rational choices in non-ambiguous trials",
            "diff_Ambg_NoAmbg_rational": "Difference in rational choice rate: ambiguous minus non-ambiguous",
            "rational_win_abg": "Proportion of rational choices in win-ambiguous trials",
            "rational_win_NoAbg": "Proportion of rational choices in win-non-ambiguous trials",
            "rational_loss_abg": "Proportion of rational choices in loss-ambiguous trials",
            "rational_loss_NoAbg": "Proportion of rational choices in loss-non-ambiguous trials",
            "diff_rational_Ambg_NoAmbg_win": "Difference in rational win rate: ambiguous minus non-ambiguous",
            "diff_rational_Ambg_NoAmbg_loss": "Difference in rational loss rate: ambiguous minus non-ambiguous",
            "diff_rational_Ambg_win_loss": "Difference in rational choice rate: win minus loss in ambiguous trials",
            "diff_rational_NoAmbg_win_loss": "Difference in rational choice rate: win minus loss in non-ambiguous trials",
            "diff_rational_Ambg_win_NoAmbg_loss": "Difference in rational choice rate: win-ambiguous minus loss-non-ambiguous",
            "diff_rational_Ambg_loss_NoAmbg_win": "Difference in rational choice rate: loss-ambiguous minus win-non-ambiguous",
            # --- Rational choices (safe/risky) ---
            "rational_safe": "Proportion of rational choices when choosing safe",
            "rational_risky": "Proportion of rational choices when choosing risky",
            "diff_rational_safe_risky": "Difference in rational choice rate: safe minus risky choices",
            "rational_safe_win": "Proportion of rational choices for safe choices in win trials",
            "rational_safe_loss": "Proportion of rational choices for safe choices in loss trials",
            "rational_risky_win": "Proportion of rational choices for risky choices in win trials",
            "rational_risky_loss": "Proportion of rational choices for risky choices in loss trials",
            "diff_rational_safe_win_risky_win": "Difference in rational win rate: safe minus risky choices",
            "diff_rational_safe_loss_risky_loss": "Difference in rational loss rate: safe minus risky choices",
            "diff_rational_safe_win_risky_loss": "Difference in rational choice rate: safe-win minus risky-loss",
            "diff_rational_safe_loss_risky_win": "Difference in rational choice rate: safe-loss minus risky-win",
            "rational_safe_abg": "Proportion of rational choices for safe choices in ambiguous trials",
            "rational_safe_NoAbg": "Proportion of rational choices for safe choices in non-ambiguous trials",
            "rational_risky_abg": "Proportion of rational choices for risky choices in ambiguous trials",
            "rational_risky_NoAbg": "Proportion of rational choices for risky choices in non-ambiguous trials",
            "diff_rational_safe_Ambg_NoAmbg": "Difference in rational safe rate: ambiguous minus non-ambiguous",
            "diff_rational_risky_Ambg_NoAmbg": "Difference in rational risky rate: ambiguous minus non-ambiguous",
            "diff_rational_safe_risky_Ambg": "Difference in rational choice rate: safe minus risky in ambiguous trials",
            "diff_rational_safe_risky_NoAmbg": "Difference in rational choice rate: safe minus risky in non-ambiguous trials",
            # --- Quality / exclusion metrics ---
            "nb_incorrect_gain": "Number of incorrect choices in no-brainer gain catch trials",
            "above_chance": "Proportion of trials where choice was above chance level",
            "chosen_prop_LeftRight": "Proportion of trials where the same left/right response was repeated",
            "chosen_prop_SafeRisky": "Proportion of trials where the same safe/risky choice was repeated",
        }

    def metrics(self):
        """
        Compute the metrics of each participant: response times, points, and the proportions of risky and rational choices, overall and by win/loss and ambiguous/non-ambiguous trials.

        Returns
        -------
        pandas.DataFrame
            The metrics, one row per participant, also stored in `results`. The columns are described in `codebook`.

        Notes
        -----
        Metrics of an empty selection, such as the proportion of risky choices in ambiguous loss trials for a participant who played none, are NaN.
        """
        rows = []
        with quiet_empty_slices():
            for user_id, user_data in self.data.groupby("userID"):
                rows.append({"userID": user_id, **_participant_metrics(user_data)})
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

        Raises
        ------
        KeyError
            If the data have no ``config_id`` column.

        Notes
        -----
        A participant is excluded if

        - they repeated the same left/right response on at least 95% of the trials,
        - they failed 2 or more of the no-brainer catch trials,
        - they played more than 40 trials (due to a technical error),
        - they played the version of the game with 60 trials (``config_id`` 242).
        """
        require(self.data, ["config_id"], "Scavenger.clean_data")
        if self.results.empty:
            self.metrics()
        results = self.results
        n_before = results["userID"].nunique()

        trials = self.data["userID"].value_counts()
        long_version = self.data.loc[self.data["config_id"] == 242, "userID"].unique()
        keep = results["userID"].isin(trials.index[trials <= 40])
        keep &= ~results["userID"].isin(long_version)
        keep &= results["nb_incorrect_gain"] < 2
        keep &= results["chosen_prop_LeftRight"] < 0.95

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


def _participant_metrics(data):
    """The metrics of one participant."""
    rt = data["RT"].to_numpy(dtype=float)
    points = data["outcome"].to_numpy(dtype=float)
    risky = data["risky"].to_numpy() == 1
    safe = data["risky"].to_numpy() == 0
    ambiguous = data["ambiguousTrial"].to_numpy(dtype=bool)
    loss = (data["safe_EV"] < 0).to_numpy()
    win = ~loss
    rational = data["EV_diff_chosen"].to_numpy(dtype=float) >= 0

    out = {"n_trials": len(data), **session_time(data["date"].iloc[0])}

    out["mean_RT"] = np.nanmean(rt)
    out["median_RT"] = np.nanmedian(rt)
    for name, mask in [
        ("safe", safe),
        ("risky", risky),
        ("win", win),
        ("loss", loss),
        ("ambg", ambiguous),
        ("NoAmbg", ~ambiguous),
        ("win_ambg", win & ambiguous),
        ("win_NoAmbg", win & ~ambiguous),
        ("loss_ambg", loss & ambiguous),
        ("loss_NoAmbg", loss & ~ambiguous),
    ]:
        out[f"median_RT_{name}"] = np.nanmedian(rt[mask])
    out["diff_RT_win_amg_win_noAmbg"] = out["median_RT_win_ambg"] - out["median_RT_win_NoAmbg"]
    out["diff_RT_loss_amg_loss_noAmbg"] = out["median_RT_loss_ambg"] - out["median_RT_loss_NoAmbg"]
    out["diff_RT_win_loss_ambg"] = out["median_RT_win_ambg"] - out["median_RT_loss_ambg"]
    out["diff_RT_win_loss_noAmbg"] = out["median_RT_win_NoAmbg"] - out["median_RT_loss_NoAmbg"]
    out["diff_RT_win_ambg_loss_noAmbg"] = out["median_RT_win_ambg"] - out["median_RT_loss_NoAmbg"]
    out["diff_RT_loss_ambg_win_noAmbg"] = out["median_RT_loss_ambg"] - out["median_RT_win_NoAmbg"]

    out["total_rewards"] = np.nansum(points)
    out["mean_points"] = np.nanmean(points)
    for name, mask in [
        ("safe", safe),
        ("risky", risky),
        ("ambg", ambiguous),
        ("NoAmbg", ~ambiguous),
        ("win_ambg", win & ambiguous),
        ("win_NoAmbg", win & ~ambiguous),
        ("loss_ambg", loss & ambiguous),
        ("loss_NoAmbg", loss & ~ambiguous),
    ]:
        out[f"mean_points_{name}"] = np.nanmean(points[mask])
    out["diff_points_win_amg_win_noAmbg"] = out["mean_points_win_ambg"] - out["mean_points_win_NoAmbg"]
    out["diff_points_loss_amg_loss_noAmbg"] = out["mean_points_loss_ambg"] - out["mean_points_loss_NoAmbg"]
    out["diff_points_win_loss_ambg"] = out["mean_points_win_ambg"] - out["mean_points_loss_ambg"]
    out["diff_points_win_loss_noAmbg"] = out["mean_points_win_NoAmbg"] - out["mean_points_loss_NoAmbg"]
    out["diff_points_win_ambg_loss_noAmbg"] = out["mean_points_win_ambg"] - out["mean_points_loss_NoAmbg"]
    out["diff_points_loss_ambg_win_noAmbg"] = out["mean_points_loss_ambg"] - out["mean_points_win_NoAmbg"]

    chosen = data["chosen"].to_numpy()
    out["choice_stickiness"] = int(np.sum(chosen[1:] == chosen[:-1]))

    every = np.ones(len(data), dtype=bool)
    out["risky_choices"] = proportion(risky, every)
    out["win_risk"] = proportion(risky, win)
    out["loss_risk"] = proportion(risky, loss)
    out["diff_loss_win_risk"] = out["loss_risk"] - out["win_risk"]
    out["risk_Ambg"] = proportion(risky, ambiguous)
    out["risk_NoAmbg"] = proportion(risky, ~ambiguous)
    out["diff_Ambg_NoAmbg_risk"] = out["risk_Ambg"] - out["risk_NoAmbg"]
    out["win_risk_abg"] = proportion(risky, win & ambiguous)
    out["win_risk_NoAbg"] = proportion(risky, win & ~ambiguous)
    out["loss_risk_abg"] = proportion(risky, loss & ambiguous)
    out["loss_risk_NoAbg"] = proportion(risky, loss & ~ambiguous)
    out["diff_Ambg_NoAmbg_win_risk"] = out["win_risk_abg"] - out["win_risk_NoAbg"]
    out["diff_Ambg_NoAmbg_loss_risk"] = out["loss_risk_abg"] - out["loss_risk_NoAbg"]
    out["diff_Ambg_win_loss_risk"] = out["win_risk_abg"] - out["loss_risk_abg"]
    out["diff_NoAmbg_win_loss_risk"] = out["win_risk_NoAbg"] - out["loss_risk_NoAbg"]
    out["diff_amg_win_nonAmg_loss_risk"] = out["win_risk_abg"] - out["loss_risk_NoAbg"]
    out["diff_amg_loss_nonAmg_win_risk"] = out["loss_risk_abg"] - out["win_risk_NoAbg"]

    out["rational_all"] = proportion(rational, every)
    out["rational_win"] = proportion(rational, win)
    out["rational_loss"] = proportion(rational, loss)
    out["diff_loss_win_rational"] = out["rational_loss"] - out["rational_win"]
    out["rational_abg"] = proportion(rational, ambiguous)
    out["rational_NoAbg"] = proportion(rational, ~ambiguous)
    out["diff_Ambg_NoAmbg_rational"] = out["rational_abg"] - out["rational_NoAbg"]
    out["rational_win_abg"] = proportion(rational, win & ambiguous)
    out["rational_win_NoAbg"] = proportion(rational, win & ~ambiguous)
    out["rational_loss_abg"] = proportion(rational, loss & ambiguous)
    out["rational_loss_NoAbg"] = proportion(rational, loss & ~ambiguous)
    out["diff_rational_Ambg_NoAmbg_win"] = out["rational_win_abg"] - out["rational_win_NoAbg"]
    out["diff_rational_Ambg_NoAmbg_loss"] = out["rational_loss_abg"] - out["rational_loss_NoAbg"]
    out["diff_rational_Ambg_win_loss"] = out["rational_win_abg"] - out["rational_loss_abg"]
    out["diff_rational_NoAmbg_win_loss"] = out["rational_win_NoAbg"] - out["rational_loss_NoAbg"]
    out["diff_rational_Ambg_win_NoAmbg_loss"] = out["rational_win_abg"] - out["rational_loss_NoAbg"]
    out["diff_rational_Ambg_loss_NoAmbg_win"] = out["rational_loss_abg"] - out["rational_win_NoAbg"]

    out["rational_safe"] = proportion(rational, safe)
    out["rational_risky"] = proportion(rational, risky)
    out["diff_rational_safe_risky"] = out["rational_safe"] - out["rational_risky"]
    out["rational_safe_win"] = proportion(rational, safe & win)
    out["rational_safe_loss"] = proportion(rational, safe & loss)
    out["rational_risky_win"] = proportion(rational, risky & win)
    out["rational_risky_loss"] = proportion(rational, risky & loss)
    out["diff_rational_safe_win_risky_win"] = out["rational_safe_win"] - out["rational_risky_win"]
    out["diff_rational_safe_loss_risky_loss"] = out["rational_safe_loss"] - out["rational_risky_loss"]
    out["diff_rational_safe_win_risky_loss"] = out["rational_safe_win"] - out["rational_risky_loss"]
    out["diff_rational_safe_loss_risky_win"] = out["rational_safe_loss"] - out["rational_risky_win"]
    out["rational_safe_abg"] = proportion(rational, safe & ambiguous)
    out["rational_safe_NoAbg"] = proportion(rational, safe & ~ambiguous)
    out["rational_risky_abg"] = proportion(rational, risky & ambiguous)
    out["rational_risky_NoAbg"] = proportion(rational, risky & ~ambiguous)
    out["diff_rational_safe_Ambg_NoAmbg"] = out["rational_safe_abg"] - out["rational_safe_NoAbg"]
    out["diff_rational_risky_Ambg_NoAmbg"] = out["rational_risky_abg"] - out["rational_risky_NoAbg"]
    out["diff_rational_safe_risky_Ambg"] = out["rational_safe_abg"] - out["rational_risky_abg"]
    out["diff_rational_safe_risky_NoAmbg"] = out["rational_safe_NoAbg"] - out["rational_risky_NoAbg"]

    out["nb_incorrect_gain"] = int((data["nbcorrect_gain"] == 0).sum())
    out["above_chance"] = np.nanmean(data["abovechance"])
    out["chosen_prop_LeftRight"] = np.nanmean(data["resp_rep"])
    out["chosen_prop_SafeRisky"] = np.nanmean(data["resp_rep_sr"])
    return out
