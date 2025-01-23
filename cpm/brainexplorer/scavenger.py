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
        >>> scavenger.codebook()

        Notes
        -----
        The columns required in data:

        - **A_EV**: expected value of choosing option A
        - **B_EV**: expected value of choosing option B
        - **A_perc**: Percent visible to user for option A
        - **B_perc**: Percent visible to user for option B
        - **A_magn**: Trial Values for option A
        - **B_magn**: Trial Values for option B
        - **EV_diff_chosen**: Expected value difference between option A and B
        - **chosen**: 1 if option A chosen, 2 if option B chosen
        - **chosen_left**: 1 if option A chosen, 0 if option B chosen
        - **t_loss_all**: 1 if loss trial, 0 if win trial
        - **t_abg_all**: 1 if ambiguous trial, 0 if non-ambiguous trial
        """

        ## check file extension
        if filepath.endswith(".csv"):
            self.data = pd.read_csv(filepath, header=0)
        elif filepath.endswith(".xlsx"):
            self.data = pd.read_excel(filepath, header=0)

        self.results = pd.DataFrame()
        self.codebook = {
            "win_risk": "Proportion of risky wins",
            "loss_risk": "Proportion of risky losses",
            "win_risk_abg": "Proportion of risky wins in ambiguous trials",
            "win_risk_NoAbg": "Proportion of risky wins in non-ambiguous trials",
            "loss_risk_abg": "Proportion of risky losses in ambiguous trials",
            "loss_risk_NoAbg": "Proportion of risky losses in non-ambiguous trials",
            "optimal_all": "Proportion of optimal choices",
            "optimal_win": "Proportion of optimal choices in wins",
            "optimal_loss": "Proportion of optimal choices in losses",
            "optimal_all_abg": "Proportion of optimal choices in ambiguous trials",
            "optimal_win_abg": "Proportion of optimal choices in wins in ambiguous trials",
            "optimal_loss_abg": "Proportion of optimal choices in losses in ambiguous trials",
        }

    def metrics(self):
        """
        Compute various descriptive statistics. See the `codebook` attribute for more information of the exact descriptive statistics we calculate.

        Parameters
        ----------
        data : pd.DataFrame
            Data from all users and trials to be processed.

        Returns
        -------
        results : pd.DataFrame
            A DataFrame containing the computed metrics.

        Warns
        -----
        UserWarning
            Provides a warning for `np.linalg.LinAlgError` in logistic regression from `statsmodels` for a given user when a singular matrix error occurs.

        """

        # Group data by userID
        grouped_data = self.data.groupby("userID")

        all_users_data = []

        # Loop through each group of user data
        for user_id, user_data in grouped_data:
            user_data = user_data.sort_values("trial")

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
            }

            # Loop through each trial in the user's data
            for _, trial_data in user_data.iterrows():
                metrics_data_trial = {}

                fields_to_process = [
                    "chosenOption_outcomes_percent",
                    "chosenOption_outcomes_userVisiblePercent",
                    "unchosenOption_outcomes_userVisiblePercent",
                    "chosenOption_outcomes_trialValue",
                    "unchosenOption_outcomes_trialValue",
                ]

                # Normalize the strings into arrays
                normalized_data = {}

                for field in fields_to_process:
                    if field in trial_data:
                        value = trial_data[field]
                        if isinstance(value, str):
                            normalized_data[field] = [
                                float(x.strip()) for x in value.split(",")
                            ]
                        else:
                            normalized_data[field] = [float(value)]

                if len(normalized_data["chosenOption_outcomes_percent"]) == 1:
                    metrics_data_trial["chosen"] = 1  # Option A chosen
                    metrics_data_trial["A_perc"] = normalized_data[
                        "chosenOption_outcomes_userVisiblePercent"
                    ]
                    metrics_data_trial["A_magn"] = normalized_data[
                        "chosenOption_outcomes_trialValue"
                    ]
                    metrics_data_trial["B_perc"] = normalized_data[
                        "unchosenOption_outcomes_userVisiblePercent"
                    ]
                    metrics_data_trial["B_magn"] = normalized_data[
                        "unchosenOption_outcomes_trialValue"
                    ]
                    if sum(metrics_data_trial["B_perc"]) != 1:
                        metrics_data_trial["ambiguous"] = 1
                    else:
                        metrics_data_trial["ambiguous"] = 0
                else:
                    metrics_data_trial["chosen"] = 2  # Option B chosen
                    metrics_data_trial["A_perc"] = normalized_data[
                        "unchosenOption_outcomes_userVisiblePercent"
                    ]
                    metrics_data_trial["A_magn"] = normalized_data[
                        "unchosenOption_outcomes_trialValue"
                    ]
                    metrics_data_trial["B_perc"] = normalized_data[
                        "chosenOption_outcomes_userVisiblePercent"
                    ]
                    metrics_data_trial["B_magn"] = normalized_data[
                        "chosenOption_outcomes_trialValue"
                    ]

                    if sum(metrics_data_trial["B_perc"]) != 1:
                        metrics_data_trial["ambiguous"] = 1
                    else:
                        metrics_data_trial["ambiguous"] = 0

                if trial_data["decision"] == "optionA":
                    metrics_data_trial["chosenLeft"] = 1
                else:
                    metrics_data_trial["chosenLeft"] = 0

                # Compute expected values
                metrics_data_trial["A_EV"] = np.dot(
                    np.array(metrics_data_trial["A_perc"]),
                    np.array(metrics_data_trial["A_magn"]),
                )
                metrics_data_trial["B_EV"] = 0.5 * np.array(
                    metrics_data_trial["B_perc"]
                ) + 0.5 * np.array(metrics_data_trial["B_magn"])
                metrics_data_trial["B_EV"] = metrics_data_trial["B_EV"][0]
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
                user_metrics["chosen_left_all"].append(metrics_data_trial["chosenLeft"])

            all_users_data.append(user_metrics)

        # iterate over all users
        for user_data in all_users_data:

            user_id = user_data["userID"]
            rsk = np.array(user_data["rsk_all"])
            t_loss = np.array(user_data["t_loss_all"])
            t_abg = np.array(user_data["t_abg_all"])
            ev_diff_chosen = np.array(user_data["EV_diff_chosen_all"], dtype=object)

            user_results = {"userID": user_id}

            # Number of risky choices
            user_results["win_risk"] = np.sum(rsk[~t_loss]) / np.sum(~t_loss)
            print(user_results["win_risk"])
            user_results["loss_risk"] = np.sum(rsk[t_loss]) / np.sum(t_loss)
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

            # Number of optimal choices
            user_results["optimal_all"] = np.sum(ev_diff_chosen > 0) / len(
                ev_diff_chosen
            )
            user_results["optimal_win"] = np.sum(ev_diff_chosen[~t_loss] > 0) / np.sum(
                ~t_loss
            )
            user_results["optimal_loss"] = np.sum(ev_diff_chosen[t_loss] > 0) / np.sum(
                t_loss
            )
            user_results["optimal_all_abg"] = np.sum(
                ev_diff_chosen[t_abg] > 0
            ) / np.sum(t_abg)
            user_results["optimal_win_abg"] = np.sum(
                ev_diff_chosen[~t_loss & t_abg] > 0
            ) / np.sum(~t_loss & t_abg)
            user_results["optimal_loss_abg"] = np.sum(
                ev_diff_chosen[t_loss & t_abg] > 0
            ) / np.sum(t_loss & t_abg)

            # logistic regression
            logit_data = user_data.copy()
            logit_data = pd.DataFrame(logit_data)

            logit_data["chosen"] = np.array(logit_data["chosen"]) - 1

            modelspec = "chosen ~ A_EV + B_EV"

            # Standardize A_EV and B_EV
            logit_data["A_EV"] = zscore(logit_data["A_EV"])
            logit_data["B_EV"] = zscore(logit_data["B_EV"])

            # Fit logistic regression model for all data
            logit_all = smf.logit(modelspec, data=logit_data).fit()

            # Create a mask for t_loss
            t_loss = logit_data["t_loss_all"]

            # Standardize A_EV and B_EV for non-loss cases
            logit_data.loc[~t_loss, "A_EV"] = zscore(logit_data.loc[~t_loss, "A_EV"])
            logit_data.loc[~t_loss, "B_EV"] = zscore(logit_data.loc[~t_loss, "B_EV"])

            # Fit logistic regression model for non-loss cases
            logit_win = smf.logit(modelspec, data=logit_data[~t_loss]).fit()

            # Standardize A_EV and B_EV for loss cases
            logit_data.loc[t_loss, "A_EV"] = zscore(logit_data.loc[t_loss, "A_EV"])
            logit_data.loc[t_loss, "B_EV"] = zscore(logit_data.loc[t_loss, "B_EV"])

            # Fit logistic regression model for loss cases
            # Catch singular matrix error
            try:
                logit_loss = smf.logit(modelspec, data=logit_data[t_loss]).fit()
            except np.linalg.LinAlgError:
                warnings.warn(
                    f"Singular matrix error for user {user_id} in loss case logistic regression. Skipping."
                )
                continue

            # Add results for each logistic regression model
            user_results[f"{'logit_all'}_itcp"] = logit_all.params.get("const", None)

            for coef_name, estimate in logit_all.params.items():
                if coef_name != "const":  # Skip intercept here as it's already added
                    user_results[f"{'logit_all'}_{coef_name}"] = estimate

            user_results[f"{'logit_win'}_itcp"] = logit_all.params.get("const", None)

            for coef_name, estimate in logit_all.params.items():
                if coef_name != "const":  # Skip intercept here as it's already added
                    user_results[f"{'logit_win'}_{coef_name}"] = estimate

            user_results[f"{'logit_loss'}_itcp"] = logit_all.params.get("const", None)

            for coef_name, estimate in logit_all.params.items():
                if coef_name != "const":  # Skip intercept here as it's already added
                    user_results[f"{'logit_loss'}_{coef_name}"] = estimate

            # Append user-level results
            self.results = pd.concat(
                [self.results, pd.DataFrame([user_results])], ignore_index=True
            )

        # Convert results to a DataFrame
        self.results = pd.DataFrame(self.results)
        return self.results

    def codebook(self):
        """
        Return a codebook describing the metrics.
        """
        return self.codebook
