import pandas as pd
import numpy as np
import warnings
from scipy.stats import zscore

class spaceObserver:
    """
    A class to analyze and compute various descriptive statistics from behavioural data from a metacognition task
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
        Load data to be analyised and turn into Pandas DataFrame

        Parameters
        ----------
        filepath : str
            The path to the CSV file to be processed. The column names in the data must subscribe to a prespecified convention, see Notes.

        Example
        ----------

        >>> spaceObserver = SpaceObserver("/example/2025-02-20_SpaceObserver_Data_short.xlsx")
        >>> spaceObserver.metrics()
        >>> spaceObserver.codebook()

        if data in modelling format: 
        
        >>> spaceObserver = SpaceObserver("/example/SpaceObserver_modelling_data.csv")
        >>> spaceObserver.metrics("modelling")
        >>> spaceObserver.codebook()

        Notes
        -----
        The columns required in data:

        - **nA**: Number if aliens of type A.
        - **nB**: Number of aliens of type B.
        - **nDiff**: Difference between the number of aliens of type A and B.
        - **correct**: TRUE if the participant made the correct choice, FALSE if incorrect.   
        - **choiceRT**: Response time for the choice.
        - **confidence**: Confidence rating.
        - **confidenceRT**: Response time for the confidence rating.
    

        Columns in the modelling format:

        - **signal**: 0 if number of aliens of type A >= number of aliens of type B, 1 if number of aliens of type A < number of aliens of type B.
        - **response**: 0 if left was chosen, 1 if right was chosen.
        - **accuracy**: 1 if the participant made the correct choice, 0 if incorrect.   
        - **RT_choice**: Response time for the choice.
        - **stimulus_intensity**: Absolute difference between the number of aliens of type A and B.
        - **confidence**: Confidence rating.
        - **confidenceRT**: Response time for the confidence rating.
        
        """

        ## check file extension
        if filepath.endswith(".csv"):
            self.data = pd.read_csv(filepath, header=0)
        elif filepath.endswith(".xlsx"):
            self.data = pd.read_excel(filepath, header=0)
            # self.data = pd.read_excel(filepath, header=0, nrows=500)

        self.results = pd.DataFrame()
        self.group_results = pd.DataFrame()
        self.codebook = {
        }

    def metrics(self, mode=None):
        """

        Returns 
        -------
        results : pd.DataFrame
            A DataFrame containing the results of the metrics.

        Variables
        ---------

        accuracy : float 
            The percentage of correct choices made by the participant. 

        mean_abs_evdiff : float
            The mean absolute difference between the number of aliens in the two groups.         
        
        evdiff_correct : float 
            The mean absolute difference between the number of aliens in the two groups for trials where the participant made the correct choice.            
        
        evdiff_incorrect : float
            The mean absolute difference between the number of aliens in the two groups for trials where the participant made the incorrect choice.      

        mean_RT : float 
            The mean response time across all trials.  

        std_RT : float 
            The standard deciation of the response time across all trials.     

        mean_confidence : float
            The mean confidence rating across all trials.

        mean_confidenceRT : float   
            The mean confidence response time across all trials.

        std_confidenceRT : float 
            The standard deciation of the confidence response time across all trials. 

        conf_corr : np.array
            The confidence ratings for trials where the participant made the correct choice.

        conf_inc : np.array
            The confidence ratings for trials where the participant made the incorrect choice.

        conf_corr_mean : float
            The mean confidence rating for trials where the participant made the correct choice.

        conf_inc_mean : float
            The mean confidence rating for trials where the participant made the incorrect choice.


        acc_per_conf : np.array
            The accuracy for each of the four confidence bins. The confidence ratings are sorted and divided into four bins, and the accuracy is calculated for each bin. 

            
        In "modelling" mode:
        -------------------

        mean_accuracy : float
            The mean accuracy across all trials.

        mean_RT : float
            The mean response time across all trials.

        mean_confidence : float
            The mean confidence rating across all trials.

        mean_confidenceRT : float
            The mean confidence response time across all trials.
        
        """
        if mode == None: 

            # Group data by userID
            grouped_data = self.data.groupby("userID")

            # Loop through each group of user data
            for user_id, user_data in grouped_data:
                user_data = user_data.sort_values("run")

                user_results = {"userID": user_id}

                correct = np.array(user_data["correct"])
                # transform false to 0 and true to 1
                correct = correct.astype(int)
                user_results["accuracy"] = sum(correct) / len(correct)

                abs_evdiff = np.array(np.abs(user_data["nDiff"]))
                user_results["mean_abs_evdiff"] = np.mean(abs_evdiff)
                
                correct_indices = np.where(correct == 1)[0]  # Indices where correct == 1
                incorrect_indices = np.where(correct != 1)[0]  # Indices where correct != 1

                # Absolute difference between the number of aliens in correct and incorrect trials
                user_results["evdiff_correct"] = np.mean(abs_evdiff[correct_indices])  
                user_results["evdiff_incorrect"] = np.mean(abs_evdiff[incorrect_indices])  
                
                confidence = np.array(user_data["confidence"])

                # mean confidence
                user_results["mean_confidence"] = np.mean(user_data["confidence"])
                # std
                user_results["std_confidence"] = np.std(user_data["confidence"])

                # mean reaction times
                user_results["mean_RT"] = np.mean(user_data["choiceRT"])
                user_results["std_RT"] = np.std(user_data["choiceRT"])  
                user_results['mean_confidenceRT'] = np.mean(user_data["confidenceRT"])
                user_results['std_confidenceRT'] = np.std(user_data["confidenceRT"])

                # confidence by performance
                conf_corr = confidence[correct == 1]
                conf_incorr = confidence[~correct != 1] 
                user_results["conf_corr_mean"] = np.mean(conf_corr)
                user_results["conf_incorr_mean"] = np.mean(conf_incorr)


                B = np.sort(confidence)
                I = np.argsort(confidence)

                # performance by confidence

                idxB = np.where(~np.isnan(B))[0]

                acc_per_conf = np.zeros(4)  # Preallocate array for accuracy per confidence bin

                for b in range(1, 5):
                    idx = np.arange((b - 1) * np.ceil(len(idxB) / 4), b * np.ceil(len(idxB) / 4), dtype=int)
                    idx = idx[idx < len(idxB)]  
                    
                    idx_co = I[idx]  
                    acc_per_conf[b - 1] = np.sum(correct[idx_co]) / len(idx_co) if len(idx_co) > 0 else np.nan

                user_results["acc_per_conf"] = acc_per_conf

                
                # Append user-level results
                self.results = pd.concat(
                    [self.results, pd.DataFrame([user_results])], ignore_index=True
                )

            # Convert results to a DataFrame
            self.results = pd.DataFrame(self.results)
            
            # calculate group means
            rt_mean = np.mean(self.results["mean_RT"])
            rt_std = np.std(self.results["mean_RT"])
            conf_mean = np.mean(self.results["mean_confidence"])
            conf_std = np.std(self.results["mean_confidence"])
            rt_conf_mean = np.mean(self.results["mean_confidenceRT"])
            acc_mean = np.mean(self.results["accuracy"])
            acc_std = np.std(self.results["accuracy"])
            conf_corr_mean = np.mean(self.results["conf_corr_mean"])
            conf_incorr_mean = np.mean(self.results["conf_incorr_mean"])
            abs_evdiff_mean = np.mean(self.results["mean_abs_evdiff"])
            evdiff_correct_mean = np.mean(self.results["evdiff_correct"])
            evdiff_incorrect_mean = np.mean(self.results["evdiff_incorrect"])

            self.group_results = pd.DataFrame({"mean_RT": rt_mean, "std_RT": rt_std, "mean_confidence": conf_mean, "std_confidence": conf_std, "mean_confidenceRT": rt_conf_mean, "mean_accuracy": acc_mean, "std_accuracy": acc_std, "mean_confidence_corr": conf_corr_mean, "mean_confidence_incorr": conf_incorr_mean, "mean_abs_evdiff": abs_evdiff_mean, "mean_evdiff_correct": evdiff_correct_mean, "mean_evdiff_incorrect": evdiff_incorrect_mean}, index=[0])
            
            return self.results, self.group_results
        
        
        elif mode == "modelling": 
            
            # Group data by subject
            grouped_data = self.data.groupby("subject")

            all_users_data = []

            # Loop through each group of user data
            for user_id, user_data in grouped_data:
                user_data = user_data.sort_values("trial")

                
                user_metrics = {
                    "userID": user_id,
                    "choice": [],
                    "nA": [],
                    "nB": [],
                    "nDiff": [],
                    "correct": [],
                    "RT": [],
                    "confidence": [],
                    "confidenceRT": [],
                    "signal": []
                }

                # Loop through each trial in the user's data
                for i, trial_data in user_data.iterrows():

                    user_metrics["correct"].append(trial_data["accuracy"])
                    user_metrics["choice"].append(trial_data["response"]) #left = 0, right = 1
                    user_metrics["signal"].append(trial_data["signal"]) #0 if nA >= nB, 1 if nA < nB
                    user_metrics["RT"].append(trial_data["RT_choice"])
                    user_metrics["confidence"].append(trial_data["confidence"])
                    user_metrics["confidenceRT"].append(trial_data["confidenceRT"])
                    user_metrics["nDiff"].append(trial_data["stimulus_intensity"]) 

                all_users_data.append(user_metrics)

            for user_data in all_users_data:

                user_results = {"userID": user_data["userID"]}

                user_results["mean_accuracy"] = np.mean(user_data["correct"])
                user_results["mean_RT"] = np.mean(user_data["RT"])
                user_results["mean_confidence"] = np.nanmean(user_data["confidence"])
                user_results["mean_confidenceRT"] = np.nanmean(user_data["confidenceRT"])

                # Append user-level results
                self.results = pd.concat(
                    [self.results, pd.DataFrame([user_results])], ignore_index=True
                )

            # Convert results to a DataFrame
            self.results = pd.DataFrame(self.results)
            
            # calculate group means
            rt_mean = np.mean(self.results["mean_RT"])
            rt_std = np.std(self.results["mean_RT"])
            conf_mean = np.mean(self.results["mean_confidence"])
            conf_std = np.std(self.results["mean_confidence"])
            acc_mean = np.mean(self.results["accuracy"])
            acc_std = np.std(self.results["accuracy"])
            self.group_results = pd.DataFrame({"mean_RT": rt_mean, "std_RT": rt_std, "mean_confidence": conf_mean, "std_confidence": conf_std, "mean_accuracy": acc_mean, "std_accuracy": acc_std}, index=[0])
            
            return self.results, self.group_results



    def codebook(self):
        """
        Return a codebook describing the metrics.
        """
        return self.codebook