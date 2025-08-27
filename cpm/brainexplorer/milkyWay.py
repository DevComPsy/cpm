import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class milkyWay: 
    """
    Class to calculate metrics for a two-arm bandit task, Milky Way & Pirate Market in BrainExplorer.

    """

    def __init__(self, filepath):
        """
        Load data to be analyised and turn into Pandas DataFrame. 
        Trial-level exclusion criteria are applied to the data.
        The results for Milky Way and Pirate Market trials are stored in separate DataFrames.

        Parameters
        ----------
        filepath : str
            The path to the CSV file to be processed. The column names in the data must subscribe to a prespecified convention, see Notes.

        Example
        ----------

        >>> milkyWay = milkyWay("/example/2025-02-20_SpaceObserver_Data_short.xlsx")
        >>> milkyWay.metrics()
        >>> milkyWay.clean_data()
        >>> results_MW = milkyWay.results_MW
        >>> results_PM = milkyWay.results_PM
        >>> milkyWay.codebook()
        >>> diff_results = milkyWay.difference_metrics()


        Notes
        -----
        The columns required in data:
        - userID: unique identifier for each participant
        - trial_type: the type of trial (Milky Way or Pirate Market)
        - date: the date and time of the trial
        - correct: whether the choice was correct (1) or incorrect (0)
        - outchosen: the outcome of the trial
        - rep: whether the choice was repeated (1) or not (0)
        - WSLS_v1: whether the choice was a win-stay/loose-shift choice for correct trials version 1 
        - WSLS_v2: whether the choice was a win-stay/loose-shift choice for correct trials version 2 

        The trial-level exclusion criteria are applied:
        - Attempts after the first attempt ("run" variable)

        The reaction times have to be corrected by the average time between the previous choice and new stimulus presentation which is 4200 ms. However, this correction can lead to negative reaction times, which may exclude some participants from the analysis.
        """
        
        self.data = pd.read_csv(filepath)

        #Separate into MW and PM data
        self.MW_data = self.data[self.data['trial_type'] == 'reward']
        self.PM_data = self.data[self.data['trial_type'] == 'punish']
        
        #Take first attempt (i.e., not neccessarily run == 1 for PM)
        min_run_per_user = self.MW_data.groupby('userID')['run'].transform('min')
        self.MW_data = self.MW_data[self.MW_data['run'] == min_run_per_user].reset_index(drop=True)
        
        min_run_per_user = self.PM_data.groupby('userID')['run'].transform('min')
        self.PM_data = self.PM_data[self.PM_data['run'] == min_run_per_user].reset_index(drop=True)               

        self.results_MW = pd.DataFrame()
        self.results_PM = pd.DataFrame()
        self.results_diff = pd.DataFrame()

        self.codebook = {
            "userID": "Unique identifier for each participant",
            "n_trials": "Number of trials completed by the participant",
            "trial_type": "Type of trial (Milky Way or Pirate Market)",
            "day_of_week": "Day of the week when the trial was conducted",
            "time": "Time when the trial was conducted",
            "time_of_day": "Time of day when the trial was conducted (morning, afternoon, evening, night)",
            "mean_RT_MW": "Mean reaction time for Milky Way trials",    
            "median_RT_MW": "Median reaction time for Milky Way trials",
            "mean_RT_PM": "Mean reaction time for Pirate Market trials",
            "median_RT_PM": "Median reaction time for Pirate Market trials",
            "accuracy_MW": "Mean accuracy for Milky Way trials",
            "accuracy_PM": "Mean accuracy for Pirate Market trials",
            "prop_same_choice_MW": "Proportion of same choice in Milky Way trials",
            "prop_same_choice_PM": "Proportion of same choice in Pirate Market trials",
            "mean_outcome_MW": "Mean outcome for Milky Way trials",
            "mean_outcome_PM": "Mean outcome for Pirate Market trials",
            "reward_diff_obt_forg_MW": "Mean difference between obtained and foregone reward for Milky Way trials",
            "reward_diff_obt_forg_PM": "Mean difference between obtained and foregone reward for Pirate Market trials",
            "prop_WSLS_1_MW": "Proportion of win-stay/loose-shift choices for correct trials version 1 for Milky Way trials",
            "prop_WSLS_2_MW": "Proportion of win-stay/loose-shift choices for correct trials version 2 for Milky Way trials",
            "prop_WSLS_1_PM": "Proportion of win-stay/loose-shift choices for correct trials version 1 for Pirate Market trials",
            "prop_WSLS_2_PM": "Proportion of win-stay/loose-shift choices for correct trials version 2 for Pirate Market trials"
        }


    def metrics(self):
        """
        Calculate the metrics for the data.

        Returns 
        -------
        results : pd.DataFrame
            A DataFrame containing the results of the metrics.

        Variables
        ----------
        - userID: unique identifier for each participant
        - n_trials: number of trials completed by the participant
        - day_of_week: day of the week of the trial
        - time: time of the trial
        - time_of_day: time of day of the trial (morning, afternoon, evening
        - trial_type: type of trial (Milky Way or Pirate Market)
        - mean_RT_MW: Mean reaction time for Milky Way trials
        - median_RT_MW: Median reaction time for Milky Way trials
        - mean_RT_PM: Mean reaction time for Pirate Market trials
        - median_RT_PM: Median reaction time for Pirate Market trials
        - accuracy_MW: Accuracy for Milky Way trials
        - accuracy_PM: Accuracy for Pirate Market trials
        - prop_same_choice_MW: Proportion of same choice in Milky Way trials
        - prop_same_choice_PM: Proportion of same choice in Pirate Market trials
        - mean_outcome_MW: Mean outcome for Milky Way trials
        - mean_outcome_PM: Mean outcome for Pirate Market trials
        - reward_diff_obt_forg_MW: Mean difference between obtained and foregone reward for Milky Way trials
        - reward_diff_obt_forg_PM: Mean difference between obtained and foregone reward for Pirate Market trials
        - prop_WSLS_1_MW: Proportion of win-stay/loose-shift choices for correct trials version 1 for Milky Way trials
        - prop_WSLS_2_MW: Proportion of win-stay/loose-shift choices for correct trials version 2 for Milky Way trials
        - prop_WSLS_1_PM: Proportion of win-stay/loose-shift choices for correct trials version 1 for Pirate Market trials
        - prop_WSLS_2_PM: Proportion of win-stay/loose-shift choices for correct trials version 2 for Pirate Market trials
        
        """
        # group MilkyWay and Pirate Market data
        MW_data = self.MW_data.groupby("userID")
        PM_data = self.PM_data.groupby("userID")

        # Loop through each group of user data
        for user_id, user_data_MW in MW_data:

            user_results_MW = {"userID": user_id}

            user_results_MW["n_trials"] = len(user_data_MW)

            user_results_MW["trial_type"] = "Milky Way"

            date = user_data_MW["date"].iloc[0]
            if isinstance(date, str):
                date = pd.to_datetime(date, format="%Y-%m-%d %H:%M:%S.%f")
            user_data_MW["day_of_week"] = date.day_name()
            user_data_MW["time"] = date.time() if hasattr(date, 'time') else date.strftime("%H:%M:%S.%f")
            # morning 6-12, afternoon 12-18, evening 18-24, night 0-6
            user_data_MW["time_of_day"] = (
                "morning" if date.hour < 12 else
                "afternoon" if date.hour < 18 else
                "evening" if date.hour < 24 else
                "night" if date.hour < 6
                else "-"
            )

            #user_results_MW["mean_RT_MW"] = np.mean(user_data_MW['rt_adj'])
            #user_results_MW["median_RT_MW"] = np.median(user_data_MW['rt_adj'])

            user_results_MW["accuracy_MW"] = np.nanmean(user_data_MW["correct"])

            user_results_MW["mean_outcome_MW"] = np.nanmean(user_data_MW["outchosen"])

            # Mean difference between obtained and foregone reward
            user_results_MW["reward_diff_obt_forg_MW"] = np.nanmean(user_data_MW["obt_min_forg"])

            # proportion of repeated choices
            user_results_MW["prop_same_choice_MW"] = np.nanmean(user_data_MW["rep"])

            # proportion of win-stay/loose-shift choices
            user_results_MW["prop_WSLS_1_MW"] = np.nanmean(
                user_data_MW["WSLS_v1"]
            )

            user_results_MW["prop_WSLS_2_MW"] = np.nanmean(
                user_data_MW["WSLS_v2"]
            )

            # Append user-level results
            self.results_MW = pd.concat(
                [self.results_MW, pd.DataFrame([user_results_MW])], ignore_index=True
            )

        self.results_MW = pd.DataFrame(self.results_MW)

        for user_id, user_data_PM in PM_data:

            user_results_PM = {"userID": user_id}

            user_results_PM["n_trials"] = len(user_data_PM)

            user_results_PM["trial_type"] = "Pirate Market"

            date = user_data_PM["date"].iloc[0]
            if isinstance(date, str):
                date = pd.to_datetime(date, format="%Y-%m-%d %H:%M:%S.%f")
            user_data_PM["day_of_week"] = date.day_name()
            user_data_PM["time"] = date.time() if hasattr(date, 'time') else date.strftime("%H:%M:%S.%f")
            # morning 6-12, afternoon 12-18, evening 18-24, night 0-6
            user_data_PM["time_of_day"] = (
                "morning" if date.hour < 12 else
                "afternoon" if date.hour < 18 else
                "evening" if date.hour < 24 else
                "night"
            )

            #user_results_PM["mean_RT_PM"] = np.mean(user_data_PM['rt_adj'])
            #user_results_PM["median_RT_PM"] = np.median(user_data_PM['rt_adj'])

            user_results_PM["accuracy_PM"] = np.nanmean(user_data_PM["correct"])

            user_results_PM["mean_outcome_PM"] = np.nanmean(user_data_PM["outchosen"])

            # Mean difference between obtained and foregone reward
            user_results_PM["reward_diff_obt_forg_PM"] = np.nanmean(user_data_PM["obt_min_forg"])

            # proportion of repeated choices
            user_results_PM["prop_same_choice_PM"] = np.nanmean(user_data_PM["rep"])

            # proportion of win-stay/loose-shift choices
            user_results_PM["prop_WSLS_1_PM"] = np.nanmean(
                user_data_PM["WSLS_v1"]
            )

            user_results_PM["prop_WSLS_2_PM"] = np.nanmean(
                user_data_PM["WSLS_v2"]
            )

            # Append user-level results
            self.results_PM = pd.concat(
                [self.results_PM, pd.DataFrame([user_results_PM])], ignore_index=True
            )

        self.results_PM = pd.DataFrame(self.results_PM)


    def difference_metrics(self):
        """
        Calculate the difference between the metrics of Milky Way and Pirate Market trials.
        
        Returns
        -------
        results_diff : pd.DataFrame
            A DataFrame containing the difference of the metrics between Milky Way and Pirate Market trials.

        Variables
        ----------
        - accuracy_diff: Difference in accuracy between Milky Way and Pirate Market trials
        - outcome_diff: Difference in mean outcome between Milky Way and Pirate Market trials
        - reward_diff_obt_forg_diff: Difference in mean difference between obtained and foregone reward between Milky Way and Pirate Market trials
        - prop_same_choice_diff: Difference in proportion of same choice between Milky Way and Pirate Market trials
        - prop_WSLS_1_diff: Difference in proportion of win-stay/loose-shift choices for correct trials version 1 between Milky Way and Pirate Market trials
        - prop_WSLS_2_diff: Difference in proportion of win-stay/loose-shift choices for correct trials version 2 between Milky Way and Pirate Market trials
        """

        # Calculate differences for each user
        for user_id in pd.unique(self.results_MW["userID"]):
            user_results_MW = self.results_MW[self.results_MW["userID"] == user_id]
            user_results_PM = self.results_PM[self.results_PM["userID"] == user_id]

            if not user_results_MW.empty and not user_results_PM.empty:
                diff_results = {
                    "userID": user_id,
                    "accuracy_diff": user_results_MW["accuracy_MW"].values[0] - user_results_PM["accuracy_PM"].values[0],
                    "outcome_diff": user_results_MW["mean_outcome_MW"].values[0] - user_results_PM["mean_outcome_PM"].values[0],
                    "reward_diff_obt_forg_diff": user_results_MW["reward_diff_obt_forg_MW"].values[0] - user_results_PM["reward_diff_obt_forg_PM"].values[0],
                    "prop_same_choice_diff": user_results_MW["prop_same_choice_MW"].values[0] - user_results_PM["prop_same_choice_PM"].values[0],
                    "prop_WSLS_1_diff": user_results_MW["prop_WSLS_1_MW"].values[0] - user_results_PM["prop_WSLS_1_PM"].values[0],
                    "prop_WSLS_2_diff": user_results_MW["prop_WSLS_2_MW"].values[0] - user_results_PM["prop_WSLS_2_PM"].values[0]
                }
                self.results_diff = pd.concat([self.results_diff, pd.DataFrame([diff_results])], ignore_index=True)

        return pd.DataFrame(self.results_diff)



    def clean_data(self):
        """
        Clean the data by removing participants who do not meet the participant-level inclusion criteria.

        Exclusion Critera
        -----------------       

        Participant-level:
        - Same choice on >= 95% of trials
        - Missing accuracy data
        - Participants with more than 72 trials (due to technical error)
        """
        nr_part_before_MW = len(self.results_MW["userID"].unique())
        nr_part_before_PM = len(self.results_PM["userID"].unique())

        #created cleaned verison of results object but also keep old object for comparison
        self.cleanedresults_MW = self.results_MW.copy()
        self.cleanedresults_PM = self.results_PM.copy()

        #Remove users with more than 72 trials 
        valid_user_ids = self.MW_data.groupby("userID")["userID"].transform("count") <= 72
        user_ids_to_keep = self.MW_data.loc[valid_user_ids, "userID"].unique()
        self.cleanedresults_MW = self.cleanedresults_MW[self.cleanedresults_MW["userID"].isin(user_ids_to_keep)]
        
        valid_user_ids = self.PM_data.groupby("userID")["userID"].transform("count") <= 72
        user_ids_to_keep = self.PM_data.loc[valid_user_ids, "userID"].unique()
        self.cleanedresults_PM = self.cleanedresults_PM[self.cleanedresults_PM["userID"].isin(user_ids_to_keep)]

        #Remove users with at least 95% same choice 
        self.cleanedresults_MW = self.cleanedresults_MW[self.cleanedresults_MW["prop_same_choice_MW"] < 0.95]
        self.cleanedresults_PM = self.cleanedresults_PM[self.cleanedresults_PM["prop_same_choice_PM"] < 0.95]
        
        #remove users with missing accuracy 
        self.cleanedresults_MW = self.cleanedresults_MW[self.cleanedresults_MW["accuracy_MW"].notna()]
        self.cleanedresults_PM = self.cleanedresults_PM[self.cleanedresults_PM["accuracy_PM"].notna()]
        
        self.deleted_participants_MW = nr_part_before_MW - len(pd.unique(self.cleanedresults_MW["userID"]))
        self.deleted_participants_PM = nr_part_before_PM - len(pd.unique(self.cleanedresults_PM["userID"]))

        #Also clean the diff file by keeping only participants who are in both the cleanedresults_MW and in the cleanedresults_PM file
        self.cleanedresults_diff = self.results_diff.copy()

        nr_part_before_diff = len(self.results_diff["userID"].unique())

        user_ids_MW = set(self.cleanedresults_MW["userID"].unique())         # Get userIDs from both cleaned datasets
        user_ids_PM = set(self.cleanedresults_PM["userID"].unique())
        valid_user_ids_diff = user_ids_MW.intersection(user_ids_PM)         # Keep only userIDs that are in both
        self.cleanedresults_diff = self.cleanedresults_diff[self.cleanedresults_diff["userID"].isin(valid_user_ids_diff)]

        self.deleted_participants_diff = nr_part_before_diff - len(self.cleanedresults_diff["userID"].unique())

        return self.cleanedresults_MW, self.cleanedresults_PM, self.cleanedresults_diff

    def codebook(self):
        """
        Create a codebook for the data.
        """        
        return self.codebook 