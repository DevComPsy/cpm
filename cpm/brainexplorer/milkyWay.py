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
        - Participants with more than 72 trials (due to technical error)
        """
        
        self.data = pd.read_csv(filepath)

        nr_part_before = len(self.data["userID"].unique())

        #trial-level exclusion criteria
        self.data = self.data[self.data["run"] == 1]  # only keep first attempt
        self.data = self.data[self.data["rt_adj"] > 150] # only keep trials with reaction time > 150 ms
        self.data = self.data[self.data["rt_adj"] < 10000] # only keep trials with reaction time < 10000 ms
        self.data = self.data[self.data.groupby("userID")["userID"].transform('count') < 72]  # only keep participants with less than 72 trials

        nr_part_after = len(self.data["userID"].unique())
        self.deleted_participants = nr_part_before - nr_part_after

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
        # seperate MilkyWay and Pirate Market data
        MW_data = self.data[self.data['trial_type'] == 'reward']
        PM_data = self.data[self.data['trial_type'] == 'punish']

        MW_data = MW_data.groupby("userID")
        PM_data = PM_data.groupby("userID")

        # Loop through each group of user data
        for user_id, user_data_MW in MW_data:

            user_results_MW = {"userID": user_id}

            user_results_MW["mean_RT_MW"] = np.mean(user_data_MW['rt_adj'])
            user_results_MW["median_RT_MW"] = np.median(user_data_MW['rt_adj'])

            user_results_MW["accuracy_MW"] = np.mean(user_data_MW["correct"])

            user_results_MW["prop_same_choice_MW"] = np.mean(user_data_MW["choice"] == 1)

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

            user_results_PM["prop_same_choice_PM"] = np.mean(user_data_PM["choice"] == 1)

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
        - Median RT > 10000 ms
        """
        combined_results = pd.concat([self.results_MW, self.results_PM], ignore_index=True)
        combined_results["userID"] = combined_results["userID"].apply(lambda x: x.iloc[0] if isinstance(x, pd.Series) else x)
        nr_part_before = len(pd.unique(combined_results["userID"]))

        self.results_MW = self.results_MW[self.results_MW["prop_same_choice_MW"] < 0.95]
        self.results_PM = self.results_PM[self.results_PM["prop_same_choice_PM"] < 0.95]

        self.results_MW = self.results_MW[self.results_MW["median_RT_MW"] < 10000]
        self.results_PM = self.results_PM[self.results_PM["median_RT_PM"] < 10000]

        combined_after = pd.concat([self.results_MW, self.results_PM], ignore_index=True)
        combined_after["userID"] = combined_after["userID"].apply(lambda x: x.iloc[0] if isinstance(x, pd.Series) else x)
        
        self.deleted_participants += nr_part_before - len(pd.unique(combined_after["userID"]))

    def codebook(self):
        """
        Create a codebook for the data.
        """        
        return self.codebook  