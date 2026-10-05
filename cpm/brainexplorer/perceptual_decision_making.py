import numpy as np
import pandas as pd

from ._utils import load_data, quiet_empty_slices, require, session_time

__all__ = ["SpaceObserver"]

COLUMNS = ["userID", "run", "Date", "accuracy", "RT_choice", "confidence", "confidenceRT", "stimulus_intensity"]


class SpaceObserver:
    """
    Compute descriptive statistics from the perceptual decision-making task with confidence judgements in BrainExplorer, *Space Observer*.

    On each trial, participants decide which of two types of alien is more numerous, and rate their confidence in their choice.
    For a more complete description of the task, see Dome, Moses-Payne, and Hauser (in prep.) or Marzuki et al. (2025).

    Parameters
    ----------
    filepath : str, os.PathLike or pandas.DataFrame
        The data, as a DataFrame or the path to a CSV or Excel (``.xlsx``) file. The column names must follow the convention in Notes.

    Attributes
    ----------
    data_raw : pandas.DataFrame
        The trials that pass the trial-level exclusion criteria.
    data_processed : pandas.DataFrame
        The metrics of each session, one row per participant and session, filled by :meth:`metrics`.
    cleanedresults : pandas.DataFrame
        The metrics of the sessions that pass the participant-level exclusion criteria, filled by :meth:`clean_data`.
    deleted_participants : int
        The number of participants that :meth:`clean_data` excluded.
    codebook : dict
        The description of each column of the metrics.

    Examples
    --------
    >>> from cpm.brainexplorer.perceptual_decision_making import SpaceObserver
    >>> observer = SpaceObserver("2025-02-20_SpaceObserver_Data.csv")
    >>> results = observer.metrics()
    >>> cleaned = observer.clean_data()
    >>> observer.get_codebook()["accuracy"]
    'Mean accuracy across all trials (proportion correct)'

    Notes
    -----
    The data must contain the following columns:

    - ``userID``: the unique identifier of the participant.
    - ``run``: the session number of the participant.
    - ``Date``: the date and time of the trial.
    - ``accuracy``: whether the choice was correct (1) or incorrect (0).
    - ``RT_choice``: the response time of the choice, in ms.
    - ``confidence``: the confidence rating.
    - ``confidenceRT``: the response time of the confidence rating, in ms.
    - ``stimulus_intensity``: the evidence strength, the difference in evidence between the two types of stimuli.

    Trials are excluded if the response time of the choice or of the confidence rating is below 150 ms or above 10000 ms, or if they have no confidence rating, since those are practice trials.
    For participant-level exclusions, see :meth:`clean_data`, and Dome, Moses-Payne, and Hauser (in prep.).

    References
    ----------
    Dome, L., Moses-Payne, M. E., & Hauser, T. U. (in prep.). Age-related shifts in metacognition reveals converging confidence for correct and error judgments across the lifespan.

    Marzuki, A., Kosina, L., Dome, L., Hewitt, S., & Hauser, T. (2025). Metacognitive antecedents to states of mental ill-health: Drops in confidence precede symptoms of OCD. *Research Square*. https://doi.org/10.21203/rs.3.rs-7544256/v1
    """

    def __init__(self, filepath=None):
        data = load_data(filepath, "SpaceObserver", na_values=["NaN", "nan"])
        require(data, COLUMNS, "SpaceObserver")

        data = data[data["RT_choice"].between(150, 10000)]
        data = data[data["confidenceRT"].between(150, 10000)]
        # trials without confidence ratings are practice trials
        confidence = data["confidence"].replace(["NaN", "nan", "NAN", ""], np.nan)
        data = data[confidence.notna()].copy()
        data["confidence"] = pd.to_numeric(data["confidence"], errors="coerce")

        self.data_raw = data
        self.data_processed = pd.DataFrame()
        self.codebook = {
            # --- Participant info ---
            "userID": "Unique identifier for each participant",
            "run": "Session number for the participant",
            "date": "Date and time of the first trial in the session",
            "n_trials": "Number of trials completed by the participant",
            "day_of_week": "Day of the week the session started",
            "time": "Clock time of the first trial",
            "time_of_day": "Time of day of the session (night: 0-6h, morning: 6-12h, afternoon: 12-18h, evening: 18-24h)",
            # --- Accuracy ---
            "accuracy": "Mean accuracy across all trials (proportion correct)",
            # --- Choice reaction time ---
            "mean_RT": "Mean choice response time across all trials (ms)",
            "median_RT": "Median choice response time across all trials (ms)",
            "median_RT_correct": "Median choice response time for correct trials (ms)",
            "median_RT_incorrect": "Median choice response time for incorrect trials (ms)",
            "diff_median_RT_correct_incorrect": "Difference in median choice RT: correct minus incorrect (ms)",
            # --- Confidence ---
            "mean_confidence": "Mean confidence rating across all trials",
            "sd_confidence": "Standard deviation of confidence ratings across all trials",
            "median_confidence": "Median confidence rating across all trials",
            "mean_confidence_correct": "Mean confidence rating for correct trials",
            "mean_confidence_incorrect": "Mean confidence rating for incorrect trials",
            "median_confidence_correct": "Median confidence rating for correct trials",
            "median_confidence_incorrect": "Median confidence rating for incorrect trials",
            "diff_median_conf_correct_incorrect": "Difference in median confidence: correct minus incorrect trials",
            "sd_confidence_correct": "Standard deviation of confidence ratings for correct trials",
            "sd_confidence_incorrect": "Standard deviation of confidence ratings for incorrect trials",
            "diff_sd_confidence_correct_incorrect": "Difference in confidence SD: correct minus incorrect trials",
            # --- Confidence reaction time ---
            "mean_confidenceRT": "Mean confidence response time across all trials (ms)",
            "median_confidenceRT": "Median confidence response time across all trials (ms)",
            "median_confidenceRT_correct": "Median confidence response time for correct trials (ms)",
            "median_confidenceRT_incorrect": "Median confidence response time for incorrect trials (ms)",
            "diff_median_confidenceRT_correct_incorrect": "Difference in median confidence RT: correct minus incorrect (ms)",
            # --- Confidence percentiles ---
            "confidence_10": "10th percentile of confidence ratings across all trials",
            "confidence_25": "25th percentile of confidence ratings across all trials",
            "confidence_75": "75th percentile of confidence ratings across all trials",
            "confidence_90": "90th percentile of confidence ratings across all trials",
            # --- Evidence strength ---
            "evidence_strength_mean": "Mean evidence strength (stimulus intensity) across all trials",
            "evidence_strength_correct_mean": "Mean evidence strength for correct trials",
            "evidence_strength_incorrect_mean": "Mean evidence strength for incorrect trials",
            "diff_evidence_strength_correct_incorrect": "Difference in mean evidence strength: correct minus incorrect trials",
            "median_ES": "Median evidence strength across all trials",
            # --- Evidence strength bins (trial-order quarters) ---
            "ES_bin_1": "Mean evidence strength for the first quarter of trials",
            "ES_bin_2": "Mean evidence strength for the second quarter of trials",
            "ES_bin_3": "Mean evidence strength for the third quarter of trials",
            "ES_bin_4": "Mean evidence strength for the fourth quarter of trials",
        }

    def metrics(self):
        """
        Compute the metrics of each session of each participant.

        Returns
        -------
        pandas.DataFrame
            The metrics, one row per participant and session, also stored in `data_processed`. The columns are described in `codebook`.

        Notes
        -----
        Correct trials are those with an ``accuracy`` of 1, and incorrect trials those with an ``accuracy`` of 0.
        Metrics of an empty selection, such as the median response time of incorrect trials when every choice was correct, are NaN.
        The evidence strength bins split the trials of a session, in the order they were played, into four parts of (nearly) equal size.
        """
        rows = []
        with quiet_empty_slices():
            for (user_id, run), user_data in self.data_raw.groupby(["userID", "run"]):
                rows.append({"userID": user_id, "run": run, **_session_metrics(user_data)})
        self.data_processed = pd.DataFrame(rows)
        return self.data_processed

    def clean_data(self):
        """
        Exclude participants with the participant-level exclusion criteria.

        Runs :meth:`metrics` first if it has not been run yet.

        Returns
        -------
        pandas.DataFrame
            The metrics of the sessions that pass the criteria, also stored in `cleanedresults`.

        Notes
        -----
        A session is excluded if

        - the mean accuracy is below 0.5,
        - the median response time of the choices is above 3000 ms,
        - the median response time of the confidence ratings is above 3000 ms,
        - the median evidence strength is above 25,
        - the median confidence is below 3 or above 97,
        - the 10th and 25th percentiles of confidence are equal, and so are the 75th and 90th percentiles.

        Participants with more than 80 trials in total, across all their sessions, are excluded entirely (due to a technical error).
        """
        if self.data_processed.empty:
            self.metrics()
        results = self.data_processed
        n_before = results["userID"].nunique()

        trials = self.data_raw.groupby("userID").size()
        keep = results["userID"].isin(trials.index[trials <= 80])
        keep &= results["accuracy"] >= 0.5
        keep &= results["median_RT"] <= 3000
        keep &= results["median_confidenceRT"] <= 3000
        keep &= results["median_ES"] <= 25
        keep &= results["median_confidence"].between(3, 97)
        flat = (results["confidence_10"] == results["confidence_25"]) & (
            results["confidence_75"] == results["confidence_90"]
        )
        keep &= ~flat

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


def _session_metrics(data):
    """The metrics of one session."""
    correct = (data["accuracy"] == 1).to_numpy()
    incorrect = (data["accuracy"] == 0).to_numpy()
    rt = data["RT_choice"].to_numpy(dtype=float)
    confidence = data["confidence"].to_numpy(dtype=float)
    confidence_rt = data["confidenceRT"].to_numpy(dtype=float)
    evidence = data["stimulus_intensity"].to_numpy(dtype=float)

    out = {"n_trials": len(data), **session_time(data["Date"].iloc[0])}
    out["accuracy"] = np.nanmean(data["accuracy"])

    out["mean_RT"] = np.nanmean(rt)
    out["median_RT"] = np.nanmedian(rt)
    out["median_RT_correct"] = np.nanmedian(rt[correct])
    out["median_RT_incorrect"] = np.nanmedian(rt[incorrect])
    out["diff_median_RT_correct_incorrect"] = out["median_RT_correct"] - out["median_RT_incorrect"]

    out["mean_confidence"] = np.nanmean(confidence)
    out["sd_confidence"] = np.nanstd(confidence)
    out["median_confidence"] = np.nanmedian(confidence)
    out["mean_confidence_correct"] = np.nanmean(confidence[correct])
    out["mean_confidence_incorrect"] = np.nanmean(confidence[incorrect])
    out["median_confidence_correct"] = np.nanmedian(confidence[correct])
    out["median_confidence_incorrect"] = np.nanmedian(confidence[incorrect])
    out["diff_median_conf_correct_incorrect"] = out["median_confidence_correct"] - out["median_confidence_incorrect"]
    out["sd_confidence_correct"] = np.nanstd(confidence[correct])
    out["sd_confidence_incorrect"] = np.nanstd(confidence[incorrect])
    out["diff_sd_confidence_correct_incorrect"] = out["sd_confidence_correct"] - out["sd_confidence_incorrect"]

    out["mean_confidenceRT"] = np.nanmean(confidence_rt)
    out["median_confidenceRT"] = np.nanmedian(confidence_rt)
    out["median_confidenceRT_correct"] = np.nanmedian(confidence_rt[correct])
    out["median_confidenceRT_incorrect"] = np.nanmedian(confidence_rt[incorrect])
    out["diff_median_confidenceRT_correct_incorrect"] = (
        out["median_confidenceRT_correct"] - out["median_confidenceRT_incorrect"]
    )

    for percentile in (10, 25, 75, 90):
        out[f"confidence_{percentile}"] = np.nanpercentile(confidence, percentile)

    out["evidence_strength_mean"] = np.nanmean(evidence)
    out["evidence_strength_correct_mean"] = np.nanmean(evidence[correct])
    out["evidence_strength_incorrect_mean"] = np.nanmean(evidence[incorrect])
    out["diff_evidence_strength_correct_incorrect"] = (
        out["evidence_strength_correct_mean"] - out["evidence_strength_incorrect_mean"]
    )
    for i, part in enumerate(np.array_split(evidence, 4), start=1):
        out[f"ES_bin_{i}"] = np.nanmean(part) if len(part) > 0 else np.nan
    out["median_ES"] = np.nanmedian(evidence)
    return out
