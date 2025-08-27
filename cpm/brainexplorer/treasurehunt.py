import pandas as pd
import numpy as np
import warnings
from scipy.stats import zscore
#import statsmodels.api as sm        
#import statsmodels.formula.api as smf


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

        >>> treasurehunt = treasurehunt("/example/2025-02-20_SpaceObserver_Data_short.xlsx")
        >>> treasurehunt.metrics()
        >>> treasurehunt.clean_data()
        >>> results = treasurehunt.results
        >>> treasurehunt.codebook()


        Notes
        -----
        The columns required in data:
        - userID: unique identifier for each participant
        - "date": the date and time of the trial
        - run: number of attempt by participant
        - outcome: the outcome of the trial
        - confidence: the confidence of the trial
        - RT: the reaction time of the trial
        - confidenceRT: the confidence reaction time of the trial
        - draws: number of draws in the trial
        - choseCurEv: whether the choice was in line with current evidence (1 = yes, 0 = no)
        - ev: the evidence in the trial as a list of integers

        Trial-level exclusion criteria:

        - attempts after the first attempt ("run" variable)
        """

        self.data = pd.read_csv(filepath, header=0)

        self.data = self.data[self.data["run"] == 1]  # only keep first attempt

        self.results = pd.DataFrame()

        self.codebook = {
            "userID": "Unique identifier for each participant",
            "n_trials": "Number of trials completed by the participant",
            "day_of_week": "Day of the week of the trial",
            "time": "Time of the trial",
            "time_of_day": "Time of day of the trial (morning, afternoon, evening, night)",
            "accuracy": "Mean accuracy (if the choice was the majority choice)",
            "mean_points": "Mean points received across all trials",
            "mean_n_draws": "Mean number of stimulus draws before making a decision",
            "sd_n_draws": "Standard deviation of the number of stimulus draws before making a decision",
            "median_RT_between_actions": "Median reaction time between actions",
            "median_confidenceRT": "Median confidence reaction time across all trials",
            "unique_draws": "Number of unique values in number of draws"
        }

    def metrics(self):
        """

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
        - accuracy: mean accuracy (if the choice was the majority choice)
        - mean_points: mean points received across all trials
        - mean_n_draws: mean number of stimulus draws before making a decision
        - sd_n_draws: standard deviation of the number of stimulus draws before making a decision
        - median_RT_between_actions: median reaction time between actions
        - median_confidenceRT: median confidence reaction time across all trials
        - unique_draws: number of unique values in number of draws
        """
        # turn string fields into list of ints
        self.data["ev"] = self.data["ev"].apply(lambda x: [int(i) for i in x.strip("[]").split(" ")])

        # Group data by userID
        grouped_data = self.data.groupby("userID")

        # Loop through each group of user data
        for user_id, user_data in grouped_data:

            user_results = {"userID": user_id}

            user_results["n_trials"] = len(user_data)

            date = user_data["date"].iloc[0]
            if isinstance(date, str):
                date = pd.to_datetime(date, format="%Y-%m-%d %H:%M:%S.%f")
            user_results["day_of_week"] = date.day_name()
            user_results["time"] = date.time() if hasattr(date, 'time') else date.strftime("%H:%M:%S.%f")
            # morning 6-12, afternoon 12-18, evening 18-24, night 0-6
            user_results["time_of_day"] = (
                "morning" if date.hour < 12 else
                "afternoon" if date.hour < 18 else
                "evening" if date.hour < 24 else
                "night" if date.hour < 6 else
                "-"
            )

            user_results["mean_points"] = np.nanmean(user_data["outcome"])   

            user_results["accuracy"] = np.nanmean(user_data["choseCurEv"]) 

            user_results["mean_confidence"] = np.nanmean(user_data["confidence"])
            user_results["sd_confidence"] = np.nanstd(user_data["confidence"])

            user_results["mean_n_draws"] = np.nanmean(user_data["draws"])
            user_results["sd_n_draws"] = np.nanstd(user_data["draws"])

            user_results["median_RT_between_actions"] = np.nanmedian(user_data["median_diffRT"])  

            user_results["median_confidenceRT"] = np.median(user_data["confidenceRT"])

            # check if < 3 unique values in number of samples
            user_results["unique_draws"] = user_data["draws"].nunique()

            # regression for influence of recent results on current choice 
            """
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
            """

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
        - < 3 unique values in number of samples
        """
        nr_part_before = len(self.results["userID"].unique())

        self.cleanedresults = self.results.copy()
        
        # Filter out participants who have less than 3 unique values for draws
        valid_users = self.data.groupby("userID").filter(
            lambda group: group["draws"].nunique() >= 3 
        )["userID"].unique()
        self.cleanedresults = self.cleanedresults[self.cleanedresults["userID"].isin(valid_users)]

        self.cleanedresults = self.cleanedresults[self.cleanedresults["mean_n_draws"] >= 2]  # only keep participants with mean number of draws >= 2
        self.cleanedresults = self.cleanedresults[self.cleanedresults["mean_n_draws"] <= 23]  # only keep participants with mean number of draws <= 23

        self.cleanedresults = self.cleanedresults[self.cleanedresults["accuracy"] > 0.8]  # only keep participants with mean choice in line with current evidence > 80%
        
        self.cleanedresults = self.cleanedresults[self.cleanedresults["unique_draws"] >= 3] # only keep participants with at least 3 unique values in number of draws

        self.deleted_participants = nr_part_before - len(self.cleanedresults["userID"].unique())

        return self.cleanedresults

    def codebook(self):
        """
        Return a codebook describing the metrics.
        """
        return self.codebook