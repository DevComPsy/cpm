import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class milkyWay: 
    """
    Class to calculate metrics for a two-arm bandit task, Milky Way & Pirate Market in BrainExplorer.

    """

    def __init__(self, filepath):
        """
        Load data to be analyised and turn into Pandas DataFrame

        Parameters
        ----------
        filepath : str
            The path to the CSV file to be processed. The column names in the data must subscribe to a prespecified convention, see Notes.

        Example
        ----------

        >>> milkyWay = milkyWay("/example/2025-02-20_SpaceObserver_Data_short.xlsx")
        >>> milkyWay.metrics()
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
        """
        
        self.data = pd.read_csv(filepath)

        self.results_MW = pd.DataFrame()
        self.results_PM = pd.DataFrame()
        self.codebook = {
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
        """
        # seperate MilkyWay and Pirate Market data
        MW_data = self.data[self.data['trial_type'] == 'reward']
        PM_data = self.data[self.data['trial_type'] == 'punish']

        MW_data = MW_data.groupby("userID")
        PM_data = PM_data.groupby("userID")

        # Loop through each group of user data
        for user_id, user_data_MW in MW_data:

            user_results_MW = {"userID": user_data_MW['userID']}

            user_results_MW["mean_RT_MW"] = np.mean(user_data_MW['rt_adj'])
            user_results_MW["median_RT_MW"] = np.median(user_data_MW['rt_adj'])

            user_results_MW["accuracy_MW"] = np.mean(user_data_MW["correct"])

            user_results_MW["prop_same_choice_MW"] = np.mean(user_data_MW["choice"] == 1)

            # Append user-level results
            self.results_MW = pd.concat(
                [self.results_MW, pd.DataFrame([user_results_MW])], ignore_index=True
            )

        self.results_MW = pd.DataFrame(self.results_MW)

        for user_id, user_data_PM in PM_data:

            user_results_PM = {"userID": user_data_PM['userID']}

            user_results_PM["mean_RT_PM"] = np.mean(user_data_PM['rt_adj'])
            user_results_PM["median_RT_PM"] = np.median(user_data_PM['rt_adj'])

            user_results_PM["accuracy_PM"] = np.mean(user_data_PM["correct"])

            user_results_PM["prop_same_choice_PM"] = np.mean(user_data_PM["choice"] == 1)

            # Append user-level results
            self.results_PM = pd.concat(
                [self.results_PM, pd.DataFrame([user_results_PM])], ignore_index=True
            )

        self.results_PM = pd.DataFrame(self.results_PM)

    def codebook(self):
        """
        Create a codebook for the data.
        """
        self.codebook = {
            "userID": "Unique identifier for each participant",
            "rt": "Reaction time of the trial",
            "choice": "Choice made by the participant",
            "outchosen": "Outcome reward of the chosen option",
            "reward1": "Reward of option 1",
            "reward2": "Reward of option 2"
        }
        
        return self.codebook
        