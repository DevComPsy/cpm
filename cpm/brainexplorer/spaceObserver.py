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

        - **userID: unique identifier for each participant
        - **date**: the date and time of the trial
        - **run**: number of attempt by participant
        - **accuracy**: the accuracy of the trial (1 for correct, 0 for incorrect)
        - **choiceRT**: the reaction time of the choice in the trial
        - **confidence**: the confidence rating of the trial
        - **confidenceRT**: the reaction time of the confidence rating in the trial
        - **stimulus_intensity**: the evidence strength of the trial, which is the difference of evidence for each group of stimuli.

        Trial-level exclusion critera:
        - Practice trials (run == 1)
        - Reaction time < 150 or > 10000 ms
        - Reaction time of confidence < 150 or > 10000 ms
        - Trials with missing confidence data
        """

        ## read data
        self.data = pd.read_csv(filepath, header=0, na_values=["NaN", "nan"])

        self.data = self.data[self.data["run"] != 1]  # exclude practice trials
        self.data = self.data[self.data["choiceRT"] >= 150]  # only keep trials with reaction time > 150 ms
        self.data = self.data[self.data["choiceRT"] <= 10000]  # only keep trials with reaction time < 10000 ms
        self.data = self.data[self.data["confidenceRT"] >= 150]  # only keep trials with confidence reaction time > 150 ms
        self.data = self.data[self.data["confidenceRT"] <= 10000]  # only keep trials with confidence reaction time < 10000 ms
        self.data = self.data[self.data["confidence"].notna()]  # only keep trials with confidence data
        
        self.data["confidence"] = self.data["confidence"].replace(
            ["NaN", "nan", "NAN", ""], pd.NA
        )

        self.data["confidence"] = pd.to_numeric(self.data["confidence"], errors='coerce')

        self.results = pd.DataFrame()
        self.codebook = {
            "userID": "Unique identifier for each participant",
            "day_of_week": "Day of the week of the trial",
            "time": "Time of the trial",
            "time_of_day": "Time of day of the trial (morning, afternoon, evening, night)",
            "accuracy": "Mean accuracy (if the choice was the majority choice)",
            "mean_RT": "Mean response time across all trials",
            "median_RT": "Median response time across all trials",
            "median_RT_correct": "Median response time for trials where the participant made the correct choice",
            "median_RT_incorrect": "Median response time for trials where the participant made the incorrect",
            "diff_median_RT_correct_incorrect": "Difference between the median response time for correct and incorrect choices",
            "mean_confidence": "Mean confidence rating across all trials",
            "sd_confidence": "Standard deviation of the confidence rating across all trials",
            "median_confidence": "Median confidence rating across all trials",
            "median_confidence_correct": "Median confidence rating for trials where the participant made the correct choice",
            "median_confidence_incorrect": "Median confidence rating for trials where the participant made the incorrect choice",
            "diff_median_conf_correct_incorrect": "Difference between the median confidence rating for correct and incorrect choices",
            "sd_confidence_correct": "Standard deviation of the confidence rating for trials where the participant made the correct choice",
            "sd_confidence_incorrect": "Standard deviation of the confidence rating for trials where the participant made the incorrect choice",
            "diff_sd_confidence_correct_incorrect": "Difference between the standard deviation of the confidence rating for correct and incorrect choices",
            "mean_confidenceRT": "Mean confidence response time across all trials",
            "median_confidenceRT": "Median confidence response time across all trials",
            "median_confidenceRT_correct": "Median confidence response time for trials where the participant made the correct choice",
            "median_confidenceRT_incorrect": "Median confidence response time for trials where the participant made the incorrect choice",
            "diff_median_confidenceRT_correct_incorrect": "Difference between the median confidence response time for correct and incorrect choices",
            "confidence_10": "10th percentile of the confidence rating across all trials",
            "confidence_25": "25th percentile of the confidence rating across all trials",
            "confidence_75": "75th percentile of the confidence rating across all trials",
            "confidence_90": "90th percentile of the confidence rating across all trials",
            "evidence_strength_mean": "Mean evidence strength across all trials",
            "evidence_strength_correct_mean": "Mean evidence strength for trials where the participant made the correct choice",
            "evidence_strength_incorrect_mean": "Mean evidence strength for trials where the participant made the incorrect choice",
            "diff_evidence_strength_correct_incorrect": "Difference between the mean evidence strength for correct and incorrect choices",
            "ES_bins": "Mean evidence strength for each of the bins. The trials here are split into 4 quarters. The evidence strength is divided into four equal bins and the mean is calculated for each bin."
        }

    def metrics(self, mode=None):
        """

        Returns 
        -------
        results : pd.DataFrame
            A DataFrame containing the results of the metrics.

        Variables
        ---------
        - userID: unique identifier for each participant
        - day_of_week: day of the week of the trial
        - time: time of the trial
        - time_of_day: time of day of the trial (morning, afternoon, evening, night)
        - accuracy : The mean accuracy across all trials.
        - mean_RT : The mean response time across all trials.
        - median_RT : The median response time across all trials.
        - median_RT_correct : The median response time for trials where the participant made the correct choice.
        - median_RT_incorrect : The median response time for trials where the participant made the incorrect choice.
        - diff_median_RT_correct_incorrect : The difference between the median response time for correct and incorrect choices.
        - mean_confidence : The mean confidence rating across all trials.
        - median_confidence : The median confidence rating across all trials.
        - median_confidence_correct : The median confidence rating for trials where the participant made the correct choice.
        - median_confidence_incorrect : The median confidence rating for trials where the participant made the incorrect choice.
        - diff_median_conf_correct_incorrect : The difference between the median confidence rating for correct and incorrect choices.
        - sd_confidence : The standard deviation of the confidence rating across all trials.
        - sd_confidence_correct : The standard deviation of the confidence rating for trials where the participant made the correct choice.   
        - sd_confidence_incorrect : The standard deviation of the confidence rating for trials where the participant made the incorrect choice.
        - diff_sd_confidence_correct_incorrect : The difference between the standard deviation of the confidence rating for correct and incorrect choices.
        - mean_confidenceRT : The mean confidence response time across all trials.
        - median_confidenceRT : The median confidence response time across all trials.
        - median_confidenceRT_correct : The median confidence response time for trials where the participant made the correct choice.
        - median_confidenceRT_incorrect : The median confidence response time for trials where the participant made the incorrect choice.
        - diff_median_confidenceRT_correct_incorrect : The difference between the median confidence response time for correct and incorrect choices.
        - confidence_10 : The 10th percentile of the confidence rating across all trials.
        - confidence_25 : The 25th percentile of the confidence rating across all trials.
        - confidence_75 : The 75th percentile of the confidence rating across all trials.
        - confidence_90 : The 90th percentile of the confidence rating across all trials.
        - evidence_strength_mean : The mean evidence strength across all trials.
        - evidence_strength_correct_mean : The mean evidence strength for trials where the participant made the correct choice.
        - evidence_strength_incorrect_mean : The mean evidence strength for trials where the participant made the incorrect choice.
        - diff_evidence_strength_correct_incorrect : The difference between the mean evidence strength for correct and incorrect choices.
        - ES_bins : The mean evidence strength for each of the bins. The trials here are split into 4 quarters. The evidence strength is divided into four equal bins and the mean is calculated for each bin.
        """            

        # Group data by subject
        grouped_data = self.data.groupby("userID")

        for user_id, user_data in grouped_data:

            user_results = {"userID": user_id}


            user_results = {"userID": user_id}

            date = user_data["date"].iloc[0]
            if isinstance(date, str):
                date = pd.to_datetime(date, format="%Y-%m-%d %H:%M:%S.%f")
            user_results["date"] = date
            user_results["day_of_week"] = date.day_name()
            user_results["time"] = date.time() if hasattr(date, 'time') else date.strftime("%H:%M:%S.%f")
            # morning 6-12, afternoon 12-18, evening 18-24, night 0-6
            user_results["time_of_day"] = (
                "morning" if date.hour < 12 else
                "afternoon" if date.hour < 18 else
                "evening" if date.hour < 24 else
                "night"
            )

            user_results["accuracy"] = np.mean(user_data["accuracy"])

            user_results["mean_RT"] = np.mean(user_data["choiceRT"])
            user_results["median_RT"] = np.median(user_data["choiceRT"])
            # different RT for correct and incorrect trials
            user_results["median_RT_correct"] = np.median(user_data["choiceRT"][user_data["accuracy"] == 1])
            user_results["median_RT_incorrect"] = np.median(user_data["choiceRT"][user_data["accuracy"] != 1])
            user_results["diff_median_RT_correct_incorrect"] = (
                user_results["median_RT_correct"] - user_results["median_RT_incorrect"]
            )

            user_results["mean_confidence"] = np.nanmean(user_data["confidence"])
            user_results["sd_confidence"] = np.nanstd(user_data["confidence"])
            user_results["median_confidence"] = np.nanmedian(user_data["confidence"])
            # confidence difference for correct and incorrect trials    
            user_results["median_confidence_correct"] = np.nanmedian(user_data["confidence"][user_data["accuracy"] == 1])
            user_results["median_confidence_incorrect"] = np.nanmedian(user_data["confidence"][user_data["accuracy"] != 1])
            user_results["diff_median_conf_correct_incorrect"] = (
                user_results["median_confidence_correct"] - user_results["median_confidence_incorrect"]
            )
            # standard deviation of confidence for correct and incorrect trials
            user_results["sd_confidence_correct"] = np.nanstd(user_data["confidence"][user_data["accuracy"] == 1])
            user_results["sd_confidence_incorrect"] = np.nanstd(user_data["confidence"][user_data["accuracy"] != 1])    
            user_results["diff_sd_confidence_correct_incorrect"] = (
                user_results["sd_confidence_correct"] - user_results["sd_confidence_incorrect"]
            )

            user_results["mean_confidenceRT"] = np.nanmean(user_data["confidenceRT"])
            user_results["median_confidenceRT"] = np.nanmedian(user_data["confidenceRT"])
            # different RT for correct and incorrect trials
            user_results["median_confidenceRT_correct"] = np.nanmedian(user_data["confidenceRT"][user_data["accuracy"] == 1])
            user_results["median_confidenceRT_incorrect"] = np.nanmedian(user_data["confidenceRT"][user_data["accuracy"] != 1])
            user_results["diff_median_confidenceRT_correct_incorrect"] = (
                user_results["median_confidenceRT_correct"] - user_results["median_confidenceRT_incorrect"]
            )

            # calculate confidence percentiles 
            confidence = user_data["confidence"]
            user_results["confidence_10"] = np.nanpercentile(confidence, 10)
            user_results["confidence_25"] = np.nanpercentile(confidence, 25)
            user_results["confidence_75"] = np.nanpercentile(confidence, 75)
            user_results["confidence_90"] = np.nanpercentile(confidence, 90)

            # calculate evidence strength 
            evidence_strength = user_data["stimulus_intensity"]

            user_results["evidence_strength_mean"] = np.nanmean(evidence_strength)

            # evidence strenth per correct and incorrect trials
            user_results["evidence_strength_correct_mean"] = np.nanmean(evidence_strength[user_data["accuracy"] == 1])
            user_results["evidence_strength_incorrect_mean"] = np.nanmean(evidence_strength[user_data["accuracy"] != 1])
            user_results["diff_evidence_strength_correct_incorrect"] = (
                user_results["evidence_strength_correct_mean"] - user_results["evidence_strength_incorrect_mean"]
            )

            # calcualte evidence strength for each bin
            bin_size = 4

            # Divide evidence strength into 4 equal bins and calculate the mean for each bin
            bins = np.array_split(evidence_strength, bin_size)
            ES_1 = np.mean(bins[0]) if len(bins[0]) > 0 else np.nan
            ES_2 = np.mean(bins[1]) if len(bins[1]) > 0 else np.nan
            ES_3 = np.mean(bins[2]) if len(bins[2]) > 0 else np.nan
            ES_4 = np.mean(bins[3]) if len(bins[3]) > 0 else np.nan
            user_results["ES_bins"] = np.array([ES_1, ES_2, ES_3, ES_4])

            # median evidence strength
            user_results["median_ES"] = np.nanmedian(evidence_strength)


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
        - Median reaction time > 3000 ms
        - Median reaction time for confidence ratings > 3000 ms
        - Median evidence strength > 25
        - Mean confidence <3 or > 97       
        - Participants with more than 80 trials (due to technical error) 
        - Exclude participants if 10th and 25th percentile of confidence is the same AND 75th and 90th percentile of confidence is the same
        """

        nr_part_before = len(self.results["userID"].unique())

        self.cleanedresults = self.results.copy()

        # exclude participants who have more than 80 trials with run == 1
        users_80trials = self.cleanedresults.groupby("userID").filter(lambda x: len(x) >= 80)["userID"].unique()
        self.cleanedresults = self.cleanedresults[self.cleanedresults["userID"].isin(users_80trials)]

        self.cleanedresults = self.cleanedresults[self.cleanedresults["mean_accuracy"] >= 0.5] # only keep participants with mean accuracy >= 50%
        self.cleanedresults = self.cleanedresults[self.cleanedresults["median_RT"] <= 3000] # only keep participants with median reaction time <= 3000 ms
        self.cleanedresults = self.cleanedresults[self.cleanedresults["median_confidenceRT"] <= 3000] # only keep participants with median confidence reaction time <= 3000 ms
        self.cleanedresults = self.cleanedresults[self.cleanedresults["median_ES"] <= 25] # only keep participants with median evidence strength <= 25
        self.cleanedresults = self.cleanedresults[self.cleanedresults["median_confidence"] <= 97] # only keep participants with mean confidence <= 97
        self.cleanedresults = self.cleanedresults[self.cleanedresults["median_confidence"] >= 3] # only keep participants with mean confidence >= 3

        # exclude if 10th and 25th percentile of confidence is the same AND 75th and 90th percentile of confidence is the same
        self.cleanedresults = self.cleanedresults[~(
            (self.cleanedresults["confidence_10"] == self.cleanedresults["confidence_25"]) & self.cleanedresults["confidence_75"] == self.cleanedresults["confidence_90"]   
        )]

        self.deleted_participants = nr_part_before - len(self.cleanedresults["userID"].unique())

    def codebook(self):
        """
        Return a codebook describing the metrics.
        """
        return self.codebook