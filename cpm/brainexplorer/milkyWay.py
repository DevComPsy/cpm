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


        Notes
        -----
        The columns required in data:
        - userID: unique identifier for each participant
        - trial_type: the type of trial (Milky Way or Pirate Market)
        - rt_adj: the adjusted reaction time of the trial
        - choice: the choice made by the participant
        - correct: whether the choice was correct (1) or incorrect (0)

        The trial-level exclusion criteria are applied:
        - Reaction time < 150 or > 10000 ms
        - Attempts after the first attempt ("run" variable)

        The reaction times have to be corrected by the average time between the previous choice and new stimulus presentation which is 4200 ms. However, this correction can lead to negative reaction times, which may exclude some participants from the analysis.
        """
        
        self.data = pd.read_csv(filepath)

        #average time between previous choice and new stimulus presentation (including feedback on previous choice etc) is 4.2 s, so he removed that from all RTs 
        #BUT: as we will see later, this correct does generate some negative RTs and thereby excludes some participants. so think about this! 
        self.data["rt_adj_cor"] = self.data["rt_adj"] - 4200 

        #Separate into MW and PM data
        self.MW_data = self.data[self.data['trial_type'] == 'reward']
        self.PM_data = self.data[self.data['trial_type'] == 'punish']
        
        #Take first attempt (i.e., not neccessarily run == 1 for PM!)
        min_run_per_user = self.MW_data.groupby('userID')['run'].transform('min')
        self.MW_data = self.MW_data[self.MW_data['run'] == min_run_per_user].reset_index(drop=True)
        
        min_run_per_user = self.PM_data.groupby('userID')['run'].transform('min')
        self.PM_data = self.PM_data[self.PM_data['run'] == min_run_per_user].reset_index(drop=True)               

        #Apply trial-level exclusion criteria 
        self.MW_data = self.MW_data[self.MW_data["rt_adj_cor"] >= 150] # only keep trials with reaction time >= 150 ms
        self.MW_data = self.MW_data[self.MW_data["rt_adj_cor"] <= 10000] # only keep trials with reaction time <= 10000 ms
        
        self.PM_data = self.PM_data[self.PM_data["rt_adj_cor"] >= 150] # only keep trials with reaction time > 150 ms
        self.PM_data = self.PM_data[self.PM_data["rt_adj_cor"] <= 10000] # only keep trials with reaction time < 10000 ms


        self.results_MW = pd.DataFrame()
        self.results_PM = pd.DataFrame()
        self.codebook = {
            "userID": "Unique identifier for each participant",
            "trial_type": "Type of trial (Milky Way or Pirate Market)",
            "rt_adj": "Adjusted reaction time of the trial",
            "choice": "Choice made by the participant (1 or 2)",
            "correct": "Whether the choice was correct (1) or incorrect (0)",
            "mean_RT_MW": "Mean reaction time for Milky Way trials",
            "median_RT_MW": "Median reaction time for Milky Way trials",
            "mean_RT_PM": "Mean reaction time for Pirate Market trials",
            "median_RT_PM": "Median reaction time for Pirate Market trials",
            "accuracy_MW": "Accuracy for Milky Way trials",
            "accuracy_PM": "Accuracy for Pirate Market trials",
            "prop_same_choice_MW": "Proportion of same choice in Milky Way trials",
            "prop_same_choice_PM": "Proportion of same choice in Pirate Market trials",
            "prop_WSLS_1": "Proportion of win-stay/loose-shift choices for correct trials based on whether the choice was correct",
            "prop_WSLS_2": "Proportion of win-stay/loose-shift choices for correct trials based on whether participant got at least 50 points (MW) or less than 50 points (PM)"
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
        - mean_RT_MW: Mean reaction time for Milky Way trials
        - median_RT_MW: Median reaction time for Milky Way trials
        - mean_RT_PM: Mean reaction time for Pirate Market trials
        - median_RT_PM: Median reaction time for Pirate Market trials
        - accuracy_MW: Accuracy for Milky Way trials
        - accuracy_PM: Accuracy for Pirate Market trials
        - prop_same_choice_MW: Proportion of same choice in Milky Way trials
        - prop_same_choice_PM: Proportion of same choice in Pirate Market trials
        - prop_WSLS_1: Proportion of win-stay/loose-shift choices for correct trials based on whether the choice was correct
        - prop_WSLS_2: Proportion of win-stay/loose-shift choices for correct trials based on whether participant got at least 50 points (MW) or less than 50 points (PM)
        """
        # group MilkyWay and Pirate Market data
        MW_data = self.MW_data.groupby("userID")
        PM_data = self.PM_data.groupby("userID")

        # Loop through each group of user data
        for user_id, user_data_MW in MW_data:

            user_results_MW = {"userID": user_id}

            user_results_MW["mean_RT_MW"] = np.mean(user_data_MW['rt_adj'])
            user_results_MW["median_RT_MW"] = np.median(user_data_MW['rt_adj'])

            user_results_MW["accuracy_MW"] = np.mean(user_data_MW["correct"])

            user_results_MW["prop_choice1_MW"] = np.mean(user_data_MW["choice"] == 1)

            # proportion of win-stay/loose-shift choices
            user_results_MW["prop_WSLS_1"] = np.mean(
                user_data_MW["WSLS_v1"]
            )

            user_results_MW["prop_WSLS_2"] = np.mean(
                user_data_MW["WSLS_v2"]
            )

            # Append user-level results
            self.results_MW = pd.concat(
                [self.results_MW, pd.DataFrame([user_results_MW])], ignore_index=True
            )

        self.results_MW = pd.DataFrame(self.results_MW)

        for user_id, user_data_PM in PM_data:

            user_results_PM = {"userID": user_id}

            user_results_PM["mean_RT_PM"] = np.mean(user_data_PM['rt_adj'])
            user_results_PM["median_RT_PM"] = np.median(user_data_PM['rt_adj'])

            user_results_PM["accuracy_PM"] = np.mean(user_data_PM["correct"])

            user_results_PM["prop_choice1_PM"] = np.mean(user_data_PM["choice"] == 1)

            # proportion of win-stay/loose-shift choices
            user_results_PM["prop_WSLS_1"] = np.mean(
                user_data_PM["WSLS_v1"]
            )

            user_results_PM["prop_WSLS_2"] = np.mean(
                user_data_PM["WSLS_v2"]
            )

            # Append user-level results
            self.results_PM = pd.concat(
                [self.results_PM, pd.DataFrame([user_results_PM])], ignore_index=True
            )

        self.results_PM = pd.DataFrame(self.results_PM)


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
        self.cleanedresults_MW = self.cleanedresults_MW[(self.cleanedresults_MW["prop_choice1_MW"] > 0.05) & (self.cleanedresults_MW["prop_choice1_MW"] < 0.95)  ]
        self.cleanedresults_PM = self.cleanedresults_PM[(self.cleanedresults_PM["prop_choice1_PM"] > 0.05) & (self.cleanedresults_PM["prop_choice1_PM"] < 0.95)  ]
        
        #remove users with missing accuracy 
        self.cleanedresults_MW = self.cleanedresults_MW[self.cleanedresults_MW["accuracy_MW"].notna()]
        self.cleanedresults_PM = self.cleanedresults_PM[self.cleanedresults_PM["accuracy_PM"].notna()]
        
        self.deleted_participants_MW = nr_part_before_MW - len(pd.unique(self.cleanedresults_MW["userID"]))
        self.deleted_participants_PM = nr_part_before_PM - len(pd.unique(self.cleanedresults_PM["userID"]))

    def codebook(self):
        """
        Create a codebook for the data.
        """        
        return self.codebook 