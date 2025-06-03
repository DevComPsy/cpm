import pandas as pd
import numpy as np
import warnings
from scipy.stats import zscore
import statsmodels.api as sm        
import statsmodels.formula.api as smf


class treasurehunt:
    """ 
    Class to calculate metrics for the Treasure Hunt task. Includes regresssion model to predict the influence of previous evidence on current choice.
    
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
        Load data to be analyised and turn into Pandas DataFrame

        Parameters
        ----------
        filepath : str
            The path to the CSV file to be processed. The column names in the data must subscribe to a prespecified convention, see Notes.

        Example
        ----------

        >>> spaceObserver = SpaceObserver("/example/2025-02-20_SpaceObserver_Data_short.xlsx")
        >>> spaceObserver.metrics()
        >>> spaceObserver.clean_data()
        >>> results = spaceObserver.results
        >>> spaceObserver.codebook()


        Notes
        -----
        The columns required in data:
        - userID: unique identifier for each participant
        - run: number of attempt by participant
        - outcome: the outcome of the trial
        - confidence: the confidence of the trial
        - RT: the reaction time of the trial
        - confidenceRT: the confidence reaction time of the trial
        - ev: the evidence of the trial

        Trial-level exclusion criteria:

        - attempts after the first attempt ("run" variable)
        - < 3 unique values in confidence rating 
        - < 3 unique values in number of samples
        """

        self.data = pd.read_csv(filepath, header=0)

        nr_part_before = len(self.data["userID"].unique())

        self.data = self.data[self.data["run"] == 1]  # only keep first attempt
        self.data = self.data[self.data["confidence"].nunique() > 3] # only keep trials with more than 3 unique values in confidence rating
        self.data = self.data[self.data["draws"].nunique() > 3]  # only keep trials with more than 3 unique values in number of samples

        nr_part_after = len(self.data["userID"].unique())

        self.deleted_participants = nr_part_before - nr_part_after

        self.results = pd.DataFrame()

        self.codebook = {
            "userID": "Unique identifier for each participant",
            "mean_accuracy": "Mean accuracy across all trials",
            "mean_points": "Mean points received across all trials",
            "mean_confidence": "Mean confidence rating across all trials",
            "mean_RTuntildec": "Mean reaction time until decision across all trials",
            "median_RTuntildec": "Median reaction time until decision across all trials",
            "mean_diffRT": "Mean difference in reaction time between the last two stimulus presentations",
            "median_diffRT": "Median difference in reaction time between the last two stimulus presentations",
            "mean_lastRT": "Mean reaction time after the last stimulus presentation",
            "median_lastRT": "Median reaction time after the last stimulus presentation",
            "mean_confidenceRT": "Mean confidence reaction time across all trials",
            "median_confidenceRT": "Median confidence reaction time across all trials",
            "mean_n_draws": "Mean number of stimulus draws before making a decision",
            "totevminus_coef": "Coefficient for the previous total evidence in the regression model",
            "deltaev_coef": "Coefficient for the change in total evidence in the regression model"  
        }

    def metrics(self, mode=None):
        """

        Returns 
        -------
        results : pd.DataFrame
            A DataFrame containing the results of the metrics.

        Variables
        ----------
        - mean_accuracy: mean accuracy (if the choice was the majority choice)
        - mean_points: mean points received
        - mean_confidence: mean confidence of the choice
        - mean_RTuntildec: mean reaction time until decision
        - median_RTuntildec: median reaction time until decision
        - mean_diffRT: mean difference in reaction time between the last two stimulus presentations
        - median_diffRT: median difference in reaction time between the last two stimulus presentations
        - mean_lastRT: mean reaction time after the last stimulus presentation
        - median_lastRT: median reaction time after the last stimulus presentation
        - mean_confidenceRT: mean confidence reaction time
        - median_confidenceRT: median confidence reaction time
        - n_draws: mean number of stimulus draws before making a decision
        - totevminus_coef: coefficient for the previous total evidence in the regression model
        - deltaev_coef: coefficient for the change in total evidence in the regression model
        """

        # Group data by userID
        grouped_data = self.data.groupby("userID")

        # turn string fields into list of ints
        self.data["ev"] = self.data["ev"].apply(lambda x: [int(i) for i in x.strip("[]").split(" ")])

        # Loop through each group of user data
        for user_id, user_data in grouped_data:

            user_results = {"userID": user_id}

            user_results["mean_points"] = np.mean(user_data["outcome"])   

            user_results["mean_accuracy"] = np.mean(user_data["correctChoice"])

            user_results["mean_confidence"] = np.mean(user_data["confidence"])

            user_results["mean_RTuntildec"] = np.mean(user_data["RTuntildec"])
            user_results["median_RTuntildec"] = np.median(user_data["RTuntildec"])
            user_results["mean_diffRT"] = np.mean(user_data["mean_diffRT"])
            user_results["median_diffRT"] = np.median(user_data["median_diffRT"])
            user_results["mean_lastRT"] = np.mean(user_data["lastRT"])
            user_results["median_lastRT"] = np.median(user_data["lastRT"])

            user_results["mean_confidenceRT"] = np.mean(user_data["confidenceRT"])
            user_results["median_confidenceRT"] = np.median(user_data["confidenceRT"])

            user_results["mean_n_draws"] = np.mean(user_data["draws"])

            # if chocie in line with current evidence
            user_results["mean_chosEv"] = np.mean(user_results["chEv"])

            # regression for influence of recent results on current choice  

            ev = []

            for idx, row in user_data.iterrows():
                ev_sequence = row["ev"]
                
                for i in range(len(ev_sequence)):
                    if i == 0:
                        totevminus = np.nan
                        deltaev = np.nan
                    else:
                        totevminus = ev_sequence[i - 1]
                        deltaev = ev_sequence[i] - ev_sequence[i - 1]

                    cont_ch = 1 if i < len(ev_sequence) - 1 else 0  # continue vs stop

                    ev.append({
                        "totev": ev_sequence[i],
                        "totevminus": totevminus,
                        "deltaev": deltaev,
                        "cont_ch": cont_ch
                    })

            ev_reg_data = pd.DataFrame(ev)

            model_data = ev_reg_data.dropna(subset=["totevminus", "deltaev", "cont_ch"])

            if not model_data.empty:
                model = smf.glm(
                    formula="cont_ch ~ totevminus + deltaev",
                    data=model_data,
                    family=sm.families.Binomial()
                ).fit()

                user_results["totevminus_coef"] = model.params.get("totevminus", np.nan)
                user_results["deltaev_coef"] = model.params.get("deltaev", np.nan)
            else:
                user_results["totevminus_coef"] = np.nan
                user_results["deltaev_coef"] = np.nan


            # Append user-level results
            self.results = pd.concat(
                [self.results, pd.DataFrame([user_results])], ignore_index=True
            )

        # Convert results to a DataFrame
        self.results = pd.DataFrame(self.results)
        
        return self.results
    
    def clean_data(self): 
        """
        Clean the aggregated data by applying participant-level exclusion criteria.

        Exclusion criteria:
        ---------
        - Mean number of draws < 2 or > 23
        - Make choice not in line with current evidence in >= 20% of trials
        - Mean confidence < 15 or > 98

        """
        nr_part_before = len(self.results["userID"].unique())

        self.results = self.results[self.results["draws"] > 2]  # only keep participants with mean number of draws > 2
        self.results = self.results[self.results["draws"] < 23]  # only keep participants with mean number of draws < 23

        self.results = self.results[self.results["mean_confidence"] > 15]  # only keep participants with mean confidence > 15
        self.results = self.results[self.results["mean_confidence"] < 98]  # only keep participants with mean confidence < 98

        self.results = self.results[self.results["mean_chosEv"] >= 0.8]  # only keep participants with mean choice in line with current evidence >= 80%

        self.deleted_participants += nr_part_before - len(self.results["userID"].unique())

    def codebook(self):
        """
        Return a codebook describing the metrics.
        """
        return self.codebook