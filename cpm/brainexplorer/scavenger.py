import pandas as pd
import numpy as np
import warnings
from scipy.stats import zscore
import statsmodels.formula.api as smf


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

        - "choice": if option A or B chosen,
        - "risky_prob": Likelihood of winning for option B,
        - "safe_magn": Trial Values for Option A,   
        - "risky_magn1" & "risky_magn2: Trial Values for Option B,
        - "ambg": if the trial is ambiguous or not

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
            "mean_RT": "Mean reaction time",
            "median_RT": "Median reaction time",
            "total_rewards": "Total rewards earned",
            "choice_stickiness": "Number of consecutive choices for the same option",   
            "risky_choices": "Proportion of risky choices overall",
            "win_risk": "Proportion of risky wins",
            "loss_risk": "Proportion of risky losses",
            "win_risk_abg": "Proportion of risky wins in ambiguous trials",
            "win_risk_NoAbg": "Proportion of risky wins in non-ambiguous trials",
            "loss_risk_abg": "Proportion of risky losses in ambiguous trials",
            "loss_risk_NoAbg": "Proportion of risky losses in non-ambiguous trials",
            "rational_all": "Proportion of rational choices",
            "rational_win": "Proportion of rational choices in wins",
            "rational_loss": "Proportion of rational choices in losses",
            "rational_all_abg": "Proportion of rational choices in ambiguous trials",
            "rational_win_abg": "Proportion of rational choices in wins in ambiguous trials",
            "rational_loss_abg": "Proportion of rational choices in losses in ambiguous trials",
            "rational_all_NoAbg": "Proportion of rational choices in non-ambiguous trials", 
            "rational_win_NoAbg": "Proportion of rational choices in wins in non-ambiguous trials",
            "rational_loss_NoAbg": "Proportion of rational choices in losses in non-ambiguous trials",
            "rational_catchTrials": "Proportion of rational choices in catch trials",
            "above_chance": "Proportion of trials with choices above chance level",
            "chosen_left_all": "Proportion of trials where option A (safe) was chosen"
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

        mean_RT: Mean reaction time,
        median_RT: Median reaction time,
        risky_choices: Proportion of risky choices overall,
        win_risk: Proportion of wins for risky choices,
        loss_risk: Proportion of losses for risky choices, 
        risk_ambg: Proportion of risky choices in ambiguous trials,
        win_risk_abg: Proportion of risky wins in ambiguous trials, 
        win_risk_NoAbg: Proportion of risky wins in non-ambiguous trials, 
        loss_risk_abg: Proportion of risky losses in ambiguous trials,
        loss_risk_NoAbg: Proportion of risky losses in non-ambiguous trials,
        rational_all: Proportion of rational choices, 
        rational_win: Proportion of rational choices in wins,
        rational_loss: Proportion of rational choices in losses,
        rational_all_abg: Proportion of rational choices in ambiguous trials,
        rational_win_abg: Proportion of rational choices in win trials in ambiguous trials,
        rational_loss_abg: Proportion of rational choices in loss trials in ambiguous trials,
        rational_all_NoAbg: Proportion of rational choices in non-ambiguous trials,
        rational_win_NoAbg: Proportion of rational choices in win trials in non-ambiguous trials,
        rational_loss_NoAbg: Proportion of rational choices in loss trials in non-ambiguous trials
        rational_catchTrials: Proportion of rational choices in catch trials,
        above_chance: Proportion of trials with choices above chance level,
        chosen_left_all: Proportion of trials where option A (safe) was chosen.

        
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

        all_users_data = []

        # Loop through each group of user data
        for user_id, user_data in grouped_data:

            # Initialize a dictionary to store results for this user
            user_metrics = {
                "userID": user_id,
                "rsk_all": [],
                "t_loss_all": [],
                "t_abg_all": [],
                "EV_diff_chosen_all": [],
                "chosen_left_all": [],
                "A_EV": [],
                "B_EV": [],
                "chosen": [],
                "all_rewards": [],
                "choice_stickiness": [],   
                "RT": [],
                "catchtrial": [],
                "abovechance": [],
                "chosenLeft": []
            }

            # Loop through each trial in the user's data
            for _, trial_data in user_data.iterrows():
                metrics_data_trial = {}

                if trial_data["chosen"] == 0:
                    metrics_data_trial["chosen"] = 1  # Option A chosen (safe)
                    metrics_data_trial["A_perc"] = 1
                    metrics_data_trial["A_magn"] = trial_data["safe_magn"]
                    metrics_data_trial["B_perc"] = [
                        trial_data["risky_prob1"],
                        trial_data["risky_prob2"]
                    ]
                    metrics_data_trial["B_magn"] = [
                        trial_data["risky_magn1"], 
                        trial_data["risky_magn2"]
                    ]
                else:
                    metrics_data_trial["chosen"] = 2  # Option B chosen
                    metrics_data_trial["A_perc"] = 1
                    metrics_data_trial["A_magn"] = trial_data[
                        "safe_magn"
                    ]
                    metrics_data_trial["B_perc"] = [
                        trial_data["risky_prob1"],
                        trial_data["risky_prob2"]
                    ]
                    metrics_data_trial["B_magn"] = [
                        trial_data["risky_magn1"], 
                        trial_data["risky_magn2"]
                    ]

                metrics_data_trial["ambiguous"] = trial_data[
                    "ambiguousTrial"
                ]

                # Compute expected values
                metrics_data_trial["A_EV"] = np.dot(
                    np.array(metrics_data_trial["A_perc"]),
                    np.array(metrics_data_trial["A_magn"]),
                )
                metrics_data_trial["B_EV"] = np.array(
                    metrics_data_trial["B_perc"][0]
                ) * np.array(
                    metrics_data_trial["B_magn"][0]
                ) + np.array(
                    metrics_data_trial["B_perc"][1]
                ) * np.array(metrics_data_trial["B_magn"][1]) 

                metrics_data_trial["EV_diff_safe"] = (
                    metrics_data_trial["A_EV"] - metrics_data_trial["B_EV"]
                )

                if metrics_data_trial["chosen"] == 1:
                    metrics_data_trial["EV_diff_chosen"] = (
                        metrics_data_trial["A_EV"] - metrics_data_trial["B_EV"]
                    )
                elif metrics_data_trial["chosen"] == 2:
                    metrics_data_trial["EV_diff_chosen"] = (
                        metrics_data_trial["B_EV"] - metrics_data_trial["A_EV"]
                    )

                # Store the trial data for the user in the user's metrics dictionary
                user_metrics["A_EV"].append(metrics_data_trial["A_EV"])
                user_metrics["B_EV"].append(metrics_data_trial["B_EV"])
                user_metrics["chosen"].append(metrics_data_trial["chosen"])
                user_metrics["rsk_all"].append(metrics_data_trial["chosen"] - 1)
                user_metrics["t_loss_all"].append(metrics_data_trial["A_EV"] < 0)
                user_metrics["t_abg_all"].append(bool(metrics_data_trial["ambiguous"]))
                user_metrics["EV_diff_chosen_all"].append(
                    metrics_data_trial["EV_diff_chosen"]
                )
                user_metrics["RT"].append(trial_data["RT"])
                user_metrics["all_rewards"].append(trial_data["outcome"])  
                user_metrics["catchtrial"].append(trial_data["catchtrial"])
                user_metrics["abovechance"].append(trial_data["abovechance"])
                user_metrics["chosenLeft"].append(trial_data["chosenLeft"])

            all_users_data.append(user_metrics)

        # iterate over all users
        for user_data in all_users_data:

            user_id = user_data["userID"]
            rsk = np.array(user_data["rsk_all"])
            t_loss = np.array(user_data["t_loss_all"])
            t_abg = np.array(user_data["t_abg_all"])
            ev_diff_chosen = np.array(user_data["EV_diff_chosen_all"], dtype=object)
            catch_trials = np.array(user_data["catchtrial"], dtype=bool)
            abovechance = np.array(user_data["abovechance"])
            chosen_left = np.array(user_data["chosenLeft"])

            user_results = {"userID": user_id}

            user_results["mean_RT"] = np.mean(user_data["RT"])
            user_results["median_RT"] = np.median(user_data["RT"])

            user_results["total_rewards"] = np.sum(user_data["all_rewards"])

            # calculate choice stickiness
            user_results["choice_stickiness"] = np.sum(user_data["chosen"][i] == 1 and user_data["chosen"][i + 1] == 1 for i in range(len(user_data["chosen"]) - 1))

            # Number of risky choices
            user_results["risky_choices"] = np.mean(rsk == 1)
            user_results["win_risk"] = np.sum(rsk[~t_loss]) / np.sum(~t_loss)
            user_results["loss_risk"] = np.sum(rsk[t_loss]) / np.sum(t_loss)
            user_results["risk_ambg"] = np.sum(rsk[t_abg]) / np.sum(t_abg)
            user_results["win_risk_abg"] = np.sum(rsk[~t_loss & t_abg]) / np.sum(
                ~t_loss & t_abg
            )
            user_results["win_risk_NoAbg"] = np.sum(rsk[~t_loss & ~t_abg]) / np.sum(
                ~t_loss & ~t_abg
            )
            user_results["loss_risk_abg"] = np.sum(rsk[t_loss & t_abg]) / np.sum(
                t_loss & t_abg
            )
            user_results["loss_risk_NoAbg"] = np.sum(rsk[t_loss & ~t_abg]) / np.sum(
                t_loss & ~t_abg
            )

            # Number of rational choices
            user_results["rational_all"] = np.sum(ev_diff_chosen >= 0) / len(
                ev_diff_chosen
            )
            user_results["rational_win"] = np.sum(ev_diff_chosen[~t_loss] >= 0) / np.sum(
                ~t_loss
            )
            user_results["rational_loss"] = np.sum(ev_diff_chosen[t_loss] >= 0) / np.sum(
                t_loss
            )

            user_results["rational_all_abg"] = np.sum(
                ev_diff_chosen[t_abg] >= 0
            ) / np.sum(t_abg)
            user_results["rational_all_NoAbg"] = np.sum(
                ev_diff_chosen[~t_abg] >= 0
            ) / np.sum(~t_abg)

            user_results["rational_win_abg"] = np.sum(
                ev_diff_chosen[~t_loss & t_abg] >= 0
            ) / np.sum(~t_loss & t_abg)
            user_results["rational_win_NoAbg"] = np.sum(
                ev_diff_chosen[~t_loss & ~t_abg] >= 0
            ) / np.sum(~t_loss & ~t_abg)

            user_results["rational_loss_abg"] = np.sum(
                ev_diff_chosen[t_loss & t_abg] >= 0
            ) / np.sum(t_loss & t_abg)
            user_results["rational_loss_NoAbg"] = np.sum(
                ev_diff_chosen[t_loss & ~t_abg] >= 0
            ) / np.sum(t_loss & ~t_abg)

            # proportion rational chocies on catch trials
            user_results["rational_catchTrials"] = np.mean(
                ev_diff_chosen[catch_trials] >= 0
            )

            # proportion above chance level
            user_results["above_chance"] = np.sum(
                abovechance == 1
            ) / len(abovechance)

            # proportion chosen left
            user_results["chosen_left_all"] = np.sum(
                chosen_left == 1
            ) / len(chosen_left)

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

        self.cleanedresults = self.cleanedresults[self.cleanedresults["median_RT"] < 5000]
        self.cleanedresults = self.cleanedresults[self.cleanedresults["rational_catchTrials"] > 0.5]
        self.cleanedresults = self.cleanedresults[self.cleanedresults["risky_choices"] < 0.95]
        self.cleanedresults = self.cleanedresults[self.cleanedresults["chosen_left_all"] < 0.95]

        self.deleted_participants = nr_part_before - len(self.cleanedresults["userID"].unique())

    def codebook(self):
        """
        Return a codebook describing the metrics.
        """
        return self.codebook