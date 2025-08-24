import pandas as pd
import numpy as np
from scipy.stats import zscore


class Scavenger:
    """
    A class to analyze and compute various descriptive statistics from behavioural data obtained in the risky decision-making
    task implemented in BrainExplorer, Scavenger. The task requires participants to choose between two options:
    The "safe" options returns an outcome with 100% probability, whereas the "risky" option returns one of
    two potential outcomes with 50% probability each.

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

        >>> scavenger = Scavenger("/example/2025-01-10_Scavenger.xlsx")
        >>> scavenger.metrics()
        >>> scavenger.clean_data()
        >>> results = scavenger.results
        >>> results.to_csv("/example/scavenger_results.csv", index=False)
        >>> scavenger.codebook()

        Notes
        -----
        The columns required in data:

        - "run": number of attempt by participant
        - "chosen": the option chosen by the participant (1 = safe, 2 = risky)
        - "ambiguousTrial": whether the trial was ambiguous (1 = yes, 0 = no)
        - "abovechance": whether the choice was above chance level (1 = yes, 0 = no)
        - "EV_diff_chosen": the difference in expected value between the chosen and unchosen option
        - "safe_EV": the expected value of the safe option
        - "RT": the reaction time of the trial
        - "date": the date and time of the trial
        - "outcome": the outcome of the trial


        Trial-level exclusion criteria:
        - Reaction time < 150 or > 10000 ms
        - Attempts after the first attempt ("run" variable)
        """
        self.data = pd.read_csv(filepath, header=0)

        self.data = self.data[self.data["run"] == 1] # only keep first attempt
        self.data = self.data[self.data["RT"] >= 150] # only keep trials with reaction time > 150 ms
        self.data = self.data[self.data["RT"] <= 10000] # only keep trials with reaction time < 10000 ms

        self.results = pd.DataFrame()
        self.group_results = pd.DataFrame()
        self.codebook = {
                "userID": "Unique identifier for each participant",
                "day_of_week": "Day of the week of the trial",
                "time": "Time of the trial",
                "time_of_day": "Time of day of the trial (morning, afternoon, evening, night)",
                "mean_RT": "Mean reaction time",
                "median_RT": "Median reaction time",
                "median_RT_safe": "Median reaction time for safe choices",
                "median_RT_risky": "Median reaction time for risky choices",
                "median_RT_ambg": "Median reaction time for ambiguous trials",
                "median_RT_NoAmbg":"Median reaction time for non-ambiguous trials",
                "median_RT_safe_ambg": "Median reaction time for safe choices in ambiguous trials",
                "median_RT_safe_NoAmbg": "Median reaction time for safe choices in non-ambiguous trials",
                "median_RT_risky_ambg": "Median reaction time for risky choices in ambiguous trials",
                "median_RT_risky_NoAmbg": "Median reaction time for risky choices in non-ambiguous trials",
                "mean_points": "Mean points received across all trials",
                "mean_points_safe": "Mean points received for safe choices",
                "mean_points_risky": "Mean points received for risky choices",
                "mean_points_ambg": "Mean points received in ambiguous trials",
                "mean_points_NoAmbg": "Mean points received in non-ambiguous",
                "mean_points_safe_ambg": "Mean points received for safe choices in ambiguous trials",
                "mean_points_safe_NoAmbg": "Mean points received for safe choices in non-ambiguous trials",
                "mean_points_risky_ambg": "Mean points received for risky choices in ambiguous trials",
                "mean_points_risky_NoAmbg": "Mean points received for risky choices in non-ambiguous",
                "risky_choices": "Proportion of risky choices overall",
                "win_risk": "Proportion of risky wins",
                "loss_risk": "Proportion of risky losses",
                "win_risk_abg": "Proportion of risky wins in ambiguous trials",
                "win_risk_NoAbg": "Proportion of risky wins in non-ambiguous trials",
                "loss_risk_abg": "Proportion of risky losses in ambiguous trials",
                "loss_risk_NoAbg": "Proportion of risky losses in non-ambiguous trials",
                "diff_loss_win_risk": "Difference between proportion of risky losses and wins",
                "diff_Ambg_NoAmbg_risk": "Difference between proportion of risky choices in ambiguous and non-ambiguous trials",
                "diff_Ambg_NoAmbg_win_risk": "Difference between proportion of risky wins in ambiguous and non-ambiguous trials",
                "diff_Ambg_NoAmbg_loss_risk": "Difference between proportion of risky losses in ambiguous and non-ambiguous trials",
                "rational_all": "Proportion of rational choices",
                "rational_win": "Proportion of rational choices in wins",
                "rational_loss": "Proportion of rational choices in losses",
                "rational_all_abg": "Proportion of rational choices in ambiguous trials",
                "rational_win_abg": "Proportion of rational choices in wins in ambiguous trials",
                "rational_loss_abg": "Proportion of rational choices in losses in ambiguous trials",
                "rational_all_NoAbg": "Proportion of rational choices in non-ambiguous trials",
                "diff_NoAmbg_win_loss_rational": "Difference between proportion of rational choices in wins and losses in non-ambiguous trials",
                "rational_win_NoAbg": "Proportion of rational choices in wins in non-ambiguous trials",
                "rational_loss_NoAbg": "Proportion of rational choices in losses in non-ambiguous trials",
                "diff_Ambg_NoAmbg_rational": "Difference between proportion of rational choices in ambiguous and non-ambiguous trials",
                "diff_Ambg_NoAmbg_win_rational": "Difference between proportion of rational choices in wins in ambiguous and non-ambiguous trials",
                "diff_Ambg_NoAmbg_loss_rational": "Difference between proportion of rational choices in losses in ambiguous and non-ambiguous trials",
                "nb_incorrect_gain": "Number of incorrect choices in gain trials",
                "above_chance": "Proportion of trials with choices above chance level",
                "chosen_prop_LeftRight": "Proportion of trials where option A (safe) was chosen",
                "chosen_prop_SafeRisky": "Proportion of trials where the safe option was chosen"
        }

    def metrics(self):

        """
        Compute metrics from the loaded data: expected values of rewards when choosing safe vs. risky options, percentages of 
        risky and safe wins/losses as well as total rational choices. 

        Parameters
        ----------
        data : pd.DataFrame
            Data from all users and trials to be processed.

        Returns 
        -------
        results : pd.DataFrame
            A DataFrame containing the computed metrics.

        Variables
        ----------
        - userID: Unique identifier for each participant,
        - day_of_week: Day of the week of the trial,
        - time: Time of the trial,
        - time_of_day: Time of day of the trial (morning, afternoon, evening, night),
        - mean_RT: Mean reaction time,
        - median_RT: Median reaction time,
        - median_RT_safe: Median reaction time for safe choices,
        - median_RT_risky: Median reaction time for risky choices,
        - median_RT_ambg: Median reaction time for ambiguous trials,
        - median_RT_NoAmbg: Median reaction time for non-ambiguous trials,
        - median_RT_safe_ambg: Median reaction time for safe choices in ambiguous trials,
        - median_RT_safe_NoAmbg: Median reaction time for safe choices in non-ambiguous trials,
        - median_RT_risky_ambg: Median reaction time for risky choices in ambiguous trials,
        - median_RT_risky_NoAmbg: Median reaction time for risky choices in non-ambiguous trials,
        - mean_points: Mean points received across all trials,
        - mean_points_safe: Mean points received for safe choices,
        - mean_points_risky: Mean points received for risky choices,
        - mean_points_ambg: Mean points received in ambiguous trials,
        - mean_points_NoAmbg: Mean points received in non-ambiguous
        - mean_points_safe_ambg: Mean points received for safe choices in ambiguous trials,
        - mean_points_safe_NoAmbg: Mean points received for safe choices in non-ambiguous trials,
        - mean_points_risky_ambg: Mean points received for risky choices in ambiguous trials,
        - mean_points_risky_NoAmbg: Mean points received for risky choices in non-ambiguous,
        - risky_choices: Proportion of risky choices overall,
        - win_risk: Proportion of risky wins,
        - loss_risk: Proportion of risky losses,
        - win_risk_abg: Proportion of risky wins in ambiguous trials,
        - win_risk_NoAbg: Proportion of risky wins in non-ambiguous trials,
        - loss_risk_abg: Proportion of risky losses in ambiguous trials,
        - loss_risk_NoAbg: Proportion of risky losses in non-ambiguous trials,
        - diff_loss_win_risk: Difference between proportion of risky losses and wins,
        - diff_Ambg_NoAmbg_risk: Difference between proportion of risky choices in ambiguous and non-ambiguous trials,
        - diff_Ambg_NoAmbg_win_risk: Difference between proportion of risky wins in ambiguous and non-ambiguous trials,
        - diff_Ambg_NoAmbg_loss_risk: Difference between proportion of risky losses in ambiguous and non-ambiguous trials,
        - rational_all: Proportion of rational choices,
        - rational_win: Proportion of rational choices in wins,
        - rational_loss: Proportion of rational choices in losses,
        - rational_all_abg: Proportion of rational choices in ambiguous trials,
        - rational_win_abg: Proportion of rational choices in wins in ambiguous trials,
        - rational_loss_abg: Proportion of rational choices in losses in ambiguous trials,
        - rational_all_NoAbg: Proportion of rational choices in non-ambiguous trials,
        - diff_NoAmbg_win_loss_rational: Difference between proportion of rational choices in wins and losses in non-ambiguous trials,
        - rational_win_NoAbg: Proportion of rational choices in wins in non-ambiguous trials,
        - rational_loss_NoAbg: Proportion of rational choices in losses in non-ambiguous trials,
        - diff_Ambg_NoAmbg_rational: Difference between proportion of rational choices in ambiguous and non-ambiguous trials,
        - diff_Ambg_NoAmbg_win_rational: Difference between proportion of rational choices in wins in ambiguous and non-ambiguous trials,
        - diff_Ambg_NoAmbg_loss_rational: Difference between proportion of rational choices in losses in ambiguous and non-ambiguous trials,
        - nb_incorrect_gain: Number of incorrect choices in gain trials,
        - above_chance: Proportion of trials with choices above chance level,
        - chosen_prop_LeftRight: Proportion of trials where option A (safe) was chosen,
        - chosen_prop_SafeRisky: Proportion of trials where the safe option was chosen

        
        Notes
        -----

        In the game users can choose between two options with uncertain reward or loss probabilities. They reiceve information about reward probabilities for two choise options and can choose either the safe or risky option. 
        The data is split into two categories and options, A and B. Option A is choosing the safe option, i.e. the one which has more reward, which is labeled with 1 in "chosen" and Option B is choosing the risky option, labeled with 2 in "chosen". 
        Risk values are transfromed to 0 (no risk) and 1 (risky).
        
        We differentiate between the percentages of winning reward visible to the users in the task vs. actual percentages. 
        A_perc is the percent visible to the user for the safe option A, B_perc is the percent visible to the user for risky option B.
        A_magn is the trial reward or loss values for option A, B_magn is the trial reward or loss values for option B.

        If visible percentages do not add up to 1 the trial is labeled as ambiguous.

        We calculate the expected value of the safe option A and the risky option B and then use these values to calculate the loss trials in which expected values are < 0.
        Percentages of loss and win trials are calculated for risky choice trials (by objetive risk, option B) and number of rational choices are calculated for all trials, win trials and loss trials.

     

        References
        ----------

        Habicht J, Dubois M, Michely J and Hauser TU. Do propranolol and amisulpride modulate confidence in risk-taking? Wellcome Open Res 2022, 7:23.


        """

        # Group data by userID
        grouped_data = self.data.groupby("userID")

        # Loop through each group of user data
        for user_id, user_data in grouped_data:
            t_abg = np.array(user_data["ambiguousTrial"], dtype=bool)
            abovechance = np.array(user_data["abovechance"])
            ev_diff_chosen = np.array(user_data["EV_diff_chosen"], dtype=object)
            rsk = np.array(user_data["chosen"])
            t_loss = np.array(user_data["safe_EV"] < 0)
            

            user_results = {"userID": user_id}

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
                "night"
            )


            user_results["mean_RT"] = np.nanmean(user_data["RT"])
            user_results["median_RT"] = np.nanmedian(user_data["RT"])
            # mediean RT for each condition
            user_results["median_RT_safe"] = np.nanmedian(user_data[rsk == 0]["RT"])
            user_results["median_RT_risky"] = np.nanmedian(user_data[rsk == 1]["RT"])
            user_results["median_RT_ambg"] = np.nanmedian(user_data[t_abg]["RT"])
            user_results["median_RT_NoAmbg"] = np.nanmedian(user_data[~t_abg]["RT"])
            user_results["median_RT_safe_ambg"] = np.nanmedian(user_data[t_abg & (rsk == 0)]["RT"])
            user_results["median_RT_safe_NoAmbg"] = np.nanmedian(user_data[~t_abg & (rsk == 0)]["RT"])
            user_results["median_RT_risky_ambg"] = np.nanmedian(user_data[t_abg & (rsk == 1)]["RT"])
            user_results["median_RT_risky_NoAmbg"] = np.nanmedian(user_data[~t_abg & (rsk == 1)]["RT"])

            user_results["total_rewards"] = np.sum(user_data["outcome"])
            user_results["mean_points"] = np.nanmean(user_data["outcome"])

            # mean points for risky and safe choices
            user_results["mean_points_safe"] = np.nanmean(user_data[rsk == 0]["outcome"])
            user_results["mean_points_risky"] = np.nanmean(user_data[rsk == 1]["outcome"])

            # mean points for ambigous and non-ambiguous trials
            user_results["mean_points_ambg"] = np.nanmean(user_data[t_abg]["outcome"])
            user_results["mean_points_NoAmbg"] = np.nanmean(user_data[~t_abg]["outcome"])

            # mean points for risky and safe choices in ambiguous and non-ambiguous trials
            user_results["mean_points_safe_ambg"] = np.nanmean(user_data[t_abg & (rsk == 0)]["outcome"])
            user_results["mean_points_safe_NoAmbg"] = np.nanmean(user_data[~t_abg & (rsk == 0)]["outcome"])
            user_results["mean_points_risky_ambg"] = np.nanmean(user_data[t_abg & (rsk == 1)]["outcome"])
            user_results["mean_points_risky_NoAmbg"] = np.nanmean(user_data[~t_abg & (rsk == 1)]["outcome"])

            # choice stickiness      
            user_results["choice_stickiness"] = np.sum(user_data["chosen"].iloc[i] == 1 and user_data["chosen"].iloc[i + 1] == 1 
                                                       for i in range(len(user_data["chosen"]) - 1)
                                                       )

            # Number of risky choices
            user_results["risky_choices"] = np.nanmean(rsk == 1)

            user_results["win_risk"] = np.sum(rsk[~t_loss]) / np.sum(~t_loss)
            user_results["loss_risk"] = np.sum(rsk[t_loss]) / np.sum(t_loss)
            user_results["diff_loss_win_risk"] = user_results["loss_risk"] - user_results["win_risk"]

            user_results["risk_Ambg"] = np.sum(rsk[t_abg]) / np.sum(t_abg)
            user_results["risk_NoAmbg"] = np.sum(rsk[~t_abg]) / np.sum(~t_abg)
            user_results["diff_Ambg_NoAmbg_risk"] = user_results["risk_Ambg"] - user_results["risk_NoAmbg"]

            user_results["win_risk_abg"] = np.sum(rsk[~t_loss & t_abg]) / np.sum(~t_loss & t_abg)
            user_results["win_risk_NoAbg"] = np.sum(rsk[~t_loss & ~t_abg]) / np.sum(~t_loss & ~t_abg)
            user_results["loss_risk_abg"] = np.sum(rsk[t_loss & t_abg]) / np.sum(t_loss & t_abg)
            user_results["loss_risk_NoAbg"] = np.sum(rsk[t_loss & ~t_abg]) / np.sum(t_loss & ~t_abg)
            user_results["diff_Ambg_NoAmbg_win_risk"] = (
                user_results["win_risk_abg"] - user_results["win_risk_NoAbg"]
            )
            user_results["diff_Ambg_NoAmbg_loss_risk"] = (
                user_results["loss_risk_abg"] - user_results["loss_risk_NoAbg"]
            )
            user_results["diff_Ambg_win_loss_risk"] = (
                user_results["win_risk_abg"] - user_results["loss_risk_abg"]
            )
            user_results["diff_NoAmbg_win_loss_risk"] = (
                user_results["win_risk_NoAbg"] - user_results["loss_risk_NoAbg"]
            )

            # Number of rational choices
            user_results["rational_all"] = np.nanmean(ev_diff_chosen >= 0)

            # rational choices in win and loss trials
            user_results["rational_win"] = np.sum(ev_diff_chosen[~t_loss] >= 0) / np.sum(~t_loss)
            user_results["rational_loss"] = np.sum(ev_diff_chosen[t_loss] >= 0) / np.sum(t_loss)
            user_results["diff_loss_win_rational"] = (
                user_results["rational_loss"] - user_results["rational_win"]
            )

            # rational choices in ambiguous and non-ambiguous trials
            user_results["rational_abg"] = np.sum(ev_diff_chosen[t_abg] >= 0) / np.sum(t_abg)
            user_results["rational_NoAbg"] = np.sum(ev_diff_chosen[~t_abg] >= 0) / np.sum(~t_abg)
            user_results["diff_Ambg_NoAmbg_rational"] = (
                user_results["rational_abg"] - user_results["rational_NoAbg"]
            )
            
            # rational choices in ambiguous and non-ambiguous trials for wins and losses
            user_results["rational_win_abg"] = np.sum(ev_diff_chosen[~t_loss & t_abg] >= 0) / np.sum(~t_loss & t_abg)
            user_results["rational_win_NoAbg"] = np.sum(ev_diff_chosen[~t_loss & ~t_abg] >= 0) / np.sum(~t_loss & ~t_abg)
            user_results["rational_loss_abg"] = np.sum(ev_diff_chosen[t_loss & t_abg] >= 0) / np.sum(t_loss & t_abg)
            user_results["rational_loss_NoAbg"] = np.sum(ev_diff_chosen[t_loss & ~t_abg] >= 0) / np.sum(t_loss & ~t_abg)
            user_results["diff_rational_Ambg_NoAmbg_win"] = (
                user_results["rational_win_abg"] - user_results["rational_win_NoAbg"]
            )
            user_results["diff_rational_Ambg_NoAmbg_loss"] = (
                user_results["rational_loss_abg"] - user_results["rational_loss_NoAbg"]
            )
            user_results["diff_rational_Ambg_win_loss"] = (
                user_results["rational_win_abg"] - user_results["rational_loss_abg"]
            )
            user_results["diff_rational_NoAmbg_win_loss"] = (
                user_results["rational_win_NoAbg"] - user_results["rational_loss_NoAbg"]
            )

            # rational choices in safe and risky trials
            user_results["rational_safe"] = np.sum(ev_diff_chosen[rsk == 0] >= 0) / np.sum(rsk == 0)
            user_results["rational_risky"] = np.sum(ev_diff_chosen[rsk == 1] >= 0) / np.sum(rsk == 1)
            user_results["diff_rational_safe_risky"] = (
                user_results["rational_safe"] - user_results["rational_risky"]
            )

            # rational choices in safe and risky trials for wins and losses
            user_results["rational_safe_win"] = np.sum(ev_diff_chosen[~t_loss & rsk == 0] >= 0) / np.sum(~t_loss & rsk == 0)
            user_results["rational_safe_loss"] = np.sum(ev_diff_chosen[t_loss & rsk == 0] >= 0) / np.sum(t_loss & rsk == 0)
            user_results["rational_risky_win"] = np.sum(ev_diff_chosen[~t_loss & rsk == 1] >= 0) / np.sum(~t_loss & rsk == 1)
            user_results["rational_risky_loss"] = np.sum(ev_diff_chosen[t_loss & rsk == 1] >= 0) / np.sum(t_loss & rsk == 1)
            user_results["diff_rational_safe_win_risky_win"] = (
                user_results["rational_safe_win"] - user_results["rational_risky_win"]
            )
            user_results["diff_rational_safe_loss_risky_loss"] = (
                user_results["rational_safe_loss"] - user_results["rational_risky_loss"]
            )
            user_results["diff_rational_safe_win_risky_loss"] = (
                user_results["rational_safe_win"] - user_results["rational_risky_loss"]
            )
            user_results["diff_rational_safe_loss_risky_win"] = (
                user_results["rational_safe_loss"] - user_results["rational_risky_win"]
            )

            # rational choices in risky and safe trials for ambiguous and non-ambiguous trials
            user_results["rational_safe_abg"] = np.sum(ev_diff_chosen[t_abg & rsk == 0] >= 0) / np.sum(t_abg & rsk == 0)
            user_results["rational_safe_NoAbg"] = np.sum(ev_diff_chosen[~t_abg & rsk == 0] >= 0) / np.sum(~t_abg & rsk == 0)
            user_results["rational_risky_abg"] = np.sum(ev_diff_chosen[~t_abg & rsk == 1] >= 0) / np.sum(~t_abg & rsk == 1)
            user_results["rational_risky_NoAbg"] = np.sum(ev_diff_chosen[t_abg & rsk == 1] >= 0) / np.sum(t_abg & rsk == 1)
            user_results["diff_rational_safe_Ambg_NoAmbg"] = (
                user_results["rational_safe_abg"] - user_results["rational_safe_NoAbg"]
            )
            user_results["diff_rational_risky_Ambg_NoAmbg"] = (
                user_results["rational_risky_abg"] - user_results["rational_risky_NoAbg"]
            )
            user_results["diff_rational_safe_risky_Ambg"] = (
                user_results["rational_safe_abg"] - user_results["rational_risky_abg"]
            )
            user_results["diff_rational_safe_risky_NoAmbg"] = (
                user_results["rational_safe_NoAbg"] - user_results["rational_risky_NoAbg"]
            )

            # No Brainer Trial Choices
            # count how man wrong choices were made in catch trials
            user_results["nb_incorrect_gain"] = user_data[user_data["nbcorrect_gain"] == 0].shape[0]

            # proportion above chance level
            user_results["above_chance"] = np.nanmean(abovechance)

            # proportion of repeated choices for left/right
            user_results["chosen_prop_LeftRight"] = np.mean(user_data["resp_rep"])

            # proportion of repeated choices for safe/risky
            user_results["chosen_prop_SafeRisky"] = np.mean(user_data["resp_rep_sr"])

            # Append user-level results
            self.results = pd.concat(
                [self.results, pd.DataFrame([user_results])], ignore_index=True
            )

        # Convert results to a DataFrame
        self.results = pd.DataFrame(self.results)

        return self.results

    def clean_data(self):
        """
        Clean the data by applying participant-level exlusion criteria.

        Exclusion criteria:
        ----------
        - Median reaction time > 5000 ms
        - Same choice safe/risky on >= 95% of trials
        - Same choice left/right on >= 95% of trials
        - Fail >= 50% of catch trials
        - Participants with more than 40 trials (due to technical error)
        - Participants with configID = 242
        """
        nr_part_before = len(self.results["userID"].unique())

        self.cleanedresults = self.results.copy()

        user_id_counts = self.data["userID"].value_counts()
        user_ids_to_keep = user_id_counts[user_id_counts <= 40].index
        self.cleanedresults = self.cleanedresults[self.cleanedresults["userID"].isin(user_ids_to_keep)]

        user_ids_to_exclude = (
            self.data.loc[self.data["config_id"] == 242, "userID"].unique()
            )
        user_ids_to_keep = [uid for uid in self.data["userID"].unique() if uid not in user_ids_to_exclude]
        self.cleanedresults = self.cleanedresults[self.cleanedresults["userID"].isin(user_ids_to_keep)]

        #self.cleanedresults = self.cleanedresults[self.cleanedresults["median_RT"] < 5000]

        self.cleanedresults = self.cleanedresults[self.cleanedresults["rational_catchTrials"] > 0.5]
        self.cleanedresults = self.cleanedresults[self.cleanedresults["chosen_left_all"] < 0.95]

        self.deleted_participants = nr_part_before - len(self.cleanedresults["userID"].unique())

    def codebook(self):
        """
        Return a codebook describing the metrics.
        """
        return self.codebook