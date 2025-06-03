import pandas as pd
import numpy as np
import warnings
from scipy.stats import zscore

class spaceObserver:
    """
    A class to analyze and compute statistics from behavioural data from a metacognition task
    in BrainExplorer, Space Observer. Participants have to choose if stimulus has more of one type of alien or the other,
    and rate their confidence in their choice.

    Attributes
    ----------
    data : pd.DataFrame
        The data loaded from the CSV or Excel file.
    results : pd.DataFrame
        A DataFrame to store the computed metrics.
    codebook : dict
        A dictionary containing the codebook for the computed metrics.
    """

    def __init__(self, filepath=None):
        """
        Load data to be analyised and turn into Pandas DataFrame.
        Exlude trials not meeting the criteria.

        Parameters
        ----------
        filepath : str
            The path to the CSV file to be processed. The column names in the data must subscribe to a prespecified convention, see Notes.

        Example
        ----------

        >>> spaceObserver = SpaceObserver("/example/2025-02-20_SpaceObserver_Data.csv")
        >>> results = spaceObserver.metrics()
        >>> spaceObserver.clean_data()
        >>> results = spaceObserver.results
        >>> results.to_csv("/example/spaceObserver_results.csv", index=False)
        >>> spaceObserver.codebook()

        Notes
        -----
        The columns required in data:

        - **userID**: unique identifier for each participant
        - **round**: the round of the trial
        - **accuracy**: the accuracy of the trial (1 for correct, 0 for incorrect)
        - **choiceRT**: the reaction time of the choice in the trial
        - **confidence**: the confidence rating of the trial
        - **confidenceRT**: the reaction time of the confidence rating in the trial
        - **stimulus_intensity**: the evidence strength of the trial, which is the difference of evidence for each group of stimuli.

        Trial-level exclusion critera:
        - Attempts after the first attempt ("run" variable)
        - Reaction time < 150 or > 10000 ms
        - Reaction time of confidence < 150 or > 10000 ms
        - practice trials: if more than 8 subsequent trials where confidence data is missing, it is a practice run and will be excluded.
        - Trials with missing confidence data
        - Participants with more than 80 trials (due to technical error)
        """

        ## read data
        self.data = pd.read_csv(filepath, header=0, na_values=["NaN", "nan"])

        nr_part_before = len(self.data["userID"].unique())

        self.data = self.data[self.data["run"] == 1]  # only keep first attempt
        self.data = self.data[self.data["choiceRT"] > 150]  # only keep trials with reaction time > 150 ms
        self.data = self.data[self.data["choiceRT"] < 10000]  # only keep trials with reaction time < 10000 ms
        self.data = self.data[self.data["confidenceRT"] > 150]  # only keep trials with confidence reaction time > 150 ms
        self.data = self.data[self.data["confidenceRT"] < 10000]  # only keep trials with confidence reaction time < 10000 ms
        self.data = self.data[self.data["confidence"].notna()]  # only keep trials with confidence data
        self.data = self.data[len(self.data["userID"]) < 80]  # only keep participants with less than 80 trials

        nr_part_after = len(self.data["userID"].unique())

        self.deleted_participants = nr_part_before - nr_part_after

        # delete practice trials
        # if > 8 subseequent trials where confidence data is missing it is practice run

        self.data["confidence"] = self.data["confidence"].replace(
            ["NaN", "nan", "NAN", ""], pd.NA
        )

        self.data["confidence"] = pd.to_numeric(self.data["confidence"], errors='coerce')

        print(len(self.data))

        self.data = self.data.groupby(["userID", "round"]).filter(
            lambda run: (
                (
                    (run["confidence"].isna() != run["confidence"].isna().shift()).cumsum()
                    .where(run["confidence"].isna())
                    .value_counts()
                    .max()
                ) if run["confidence"].isna().any() else 0
            ) < 9
        )

        self.results = pd.DataFrame()
        self.codebook = {
            "userID": "Unique identifier for each participant",
            "mean_accuracy": "Mean accuracy across all trials",
            "mean_RT": "Mean response time across all trials",
            "median_RT": "Median response time across all trials",
            "mean_confidence": "Mean confidence rating across all trials",
            "mean_confidenceRT": "Mean confidence response time across all trials",
            "median_confidenceRT": "Median confidence response time across all trials",
            "mean_RT_correct": "Mean response time for correct choices",
            "median_RT_correct": "Median response time for correct choices",
            "mean_RT_incorrect": "Mean response time for incorrect choices",
            "median_RT_incorrect": "Median response time for incorrect choices",
            "conf_corr_mean": "Mean confidence rating for correct choices",
            "conf_incorr_mean": "Mean confidence rating for incorrect choices",
            "ES_bins": "Mean evidence strength for each of the bins (4 bins)"
        }

    def metrics(self, mode=None):
        """

        Returns 
        -------
        results : pd.DataFrame
            A DataFrame containing the results of the metrics.

        Variables
        ---------

        mean_accuracy : float
            The mean accuracy across all trials.

        mean_RT : float
            The mean response time across all trials.

        median_RT : float
            The median response time across all trials.

        mean_confidence : float
            The mean confidence rating across all trials.

        mean_confidenceRT : float
            The mean confidence response time across all trials.

        median_confidenceRT : float
            The median confidence response time across all trials.

        mean_RT_correct : float
            The mean response time for trials where the participant made the correct choice.

        median_RT_correct : float
            The median response time for trials where the participant made the correct choice.

        mean_RT_incorrect : float
            The mean response time for trials where the participant made the incorrect choice.

        median_RT_incorrect : float
            The median response time for trials where the participant made the incorrect choice.

        conf_corr_mean : float
            The mean confidence rating for trials where the participant made the correct choice.

        conf_incorr_mean : float
            The mean confidence rating for trials where the participant made the incorrect choice.
        
        ES_bins : np.array
            The mean evidence strength for each of the bins. The trials here are split into 4 quarters. The evidence strength is divided into four equal bins and the mean is calculated for each bin.
        """            

        # Group data by subject
        grouped_data = self.data.groupby("userID")

        for user_id, user_data in grouped_data:

            user_results = {"userID": user_id}

            user_results["mean_accuracy"] = np.mean(user_data["accuracy"])

            user_results["mean_RT"] = np.mean(user_data["choiceRT"])
            user_results["median_RT"] = np.median(user_data["choiceRT"])

            user_results["mean_confidence"] = np.nanmean(user_data["confidence"])

            user_results["mean_confidenceRT"] = np.nanmean(user_data["confidenceRT"])
            user_results["median_confidenceRT"] = np.nanmedian(user_data["confidenceRT"])

            # confidence and RT by performance
            correct = np.array(user_data["accuracy"])
            confidence = np.array(user_data["confidence"])
            
            conf_corr = confidence[(correct == 1) & ~np.isnan(confidence)]
            conf_incorr = confidence[(correct != 1) & ~np.isnan(confidence)]
            user_results["conf_corr_mean"] = np.mean(conf_corr) if len(conf_corr) > 0 else np.nan
            user_results["conf_incorr_mean"] = np.mean(conf_incorr) if len(conf_incorr) > 0 else np.nan

            user_results["mean_RT_correct"] = np.mean(np.array(user_data["choiceRT"])[correct == 1])
            user_results["median_RT_correct"] = np.median(np.array(user_data["choiceRT"])[correct == 1])
            user_results["mean_RT_incorrect"] = np.mean(np.array(user_data["choiceRT"])[correct != 1])
            user_results["median_RT_incorrect"] = np.median(np.array(user_data["choiceRT"])[correct != 1])

            evidence_strength = user_data["stimulus_intensity"]

            # calcualte evidence strength for each bin

            bin_size = 4

            # Divide evidence strength into 4 equal bins and calculate the mean for each bin
            bins = np.array_split(evidence_strength, bin_size)
            ES_1 = np.mean(bins[0]) if len(bins[0]) > 0 else np.nan
            ES_2 = np.mean(bins[1]) if len(bins[1]) > 0 else np.nan
            ES_3 = np.mean(bins[2]) if len(bins[2]) > 0 else np.nan
            ES_4 = np.mean(bins[3]) if len(bins[3]) > 0 else np.nan
            user_results["ES_bins"] = np.array([ES_1, ES_2, ES_3, ES_4])

            # Append user-level results
            self.results = pd.concat(
                [self.results, pd.DataFrame([user_results])], ignore_index=True
            )

        # Convert results to a DataFrame
        self.results = pd.DataFrame(self.results)

        return self.results

    def clean_data(self):
        """
        Clean aggregated data by applyign participant-level exclusion criteria.
        Exclusion criteria:
        ---------
        - Mean accuracy < 0.5 
        - Median reaction time > 5000 ms
        - Median reaction time for confidence ratings > 5000 ms
        - Median evidence strength > 25
        - Mean confidence > 97        
        """

        nr_part_before = len(self.results["userID"].unique())

        self.results = self.results[self.results["mean_accuracy"] >= 0.5]
        self.results = self.results[self.results["median_RT"] < 5000]
        self.results = self.results[self.results["median_confidenceRT"] < 5000]
        self.results = self.results[self.results["ES_bins"].apply(lambda x: np.nanmean(x) < 25)]
        self.results = self.results[self.results["mean_confidence"] < 97]

        self.deleted_participants += nr_part_before - len(self.results["userID"].unique())

    def codebook(self):
        """
        Return a codebook describing the metrics.
        """
        return self.codebook