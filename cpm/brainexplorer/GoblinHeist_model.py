import pandas as pd
import numpy as np
import warnings

from scipy.special import logsumexp


class GoblinHeist:
    """
    Class for the Goblin Heist task

    Attributes
    ----------
    data : pd.DataFrame
            The individual user data to be processed. The column names in the data must subscribe to a prespecified convention, see Notes.

    parameters: list, optional
        A list of parameters to be used in the model. If not specified, default parameters are used. 
        Order of parameters: inverse temperature, learning rate, eligibility trace decay, mixing weight for low stake trials, mixing weight for high stake trials, stickiness, response stickiness.

    Example
    ----------

    goblinHeist = GoblinHeist(data, [1, 0.8, 0.5, 0.8, 0.8, -0.1, -0.1])
    goblinHeist.MB_MF_model()

    Notes
    -----
    The columns required in data:
    - userID: unique identifier for each participant
    - stimuli: which planets presented in stage of the trial
    - s: the state of the trial
    - rews: the rewards for each stimulus / planet 
    - choice: the choice made by the participant
    - stake: the stake of the trial (1 or 5) 
    - points: the points received in the trial


    References
    ----------  

    Kool W, Cushman FA, Gershman SJ (2016) When Does Model-Based Control Pay Off? PLoS Comput Biol 12

    Smid, C.R., K. Ganesan, A. Thompson, R. Cañigueral, S. Veselic, J. Royer, W. Kool, T.U. Hauser, B. Bernhardt, and N. Steinbeis. “Neurocognitive Basis of Model-Based Decision Making and Its Metacontrol in Childhood.” Developmental Cognitive Neuroscience 62 (August 1, 2023): 101269.
      
    """

    def __init__(self, data, parameters=None):

        self.data = data

        if parameters is None:
            self.parameters=[1, 0.8, 0.5, 0.8, 0.8, -0.1, -0.1]
            warnings.warn("No parameters specified, using default parameters.")
        else:
            self.parameters = parameters


    def MF_MB_model(self):

        """
        Model for model-based vs model-free decision making in a decison making task


        Returns
        -------
        LL: 
            Log Likelihood 
        Q:
            Q values
        Q_MF:
            Q values for model-free learning
        Q_MB:
            Q values for model-based learning       
        """

        # pull out the paramters
        b = self.parameters[0] # softmax inverse tempterature
        lr = self.parameters[1] # learning rate
        lamb = self.parameters[2] # eligibility trace decay
        w_lo = self.parameters[3] # mixing weight for low stake trials
        w_hi = self.parameters[4] # mixing weight for high stake trials
        st = self.parameters[5] # stickiness
        respst = self.parameters[6] # response stickiness

        # initialization
        Q_MF = np.ones((2,2))*4.5
        Q2 = np.ones((2,1))*4.5
        TM = [np.eye(2), np.eye(2)] #transition matrix
        M = np.zeros((2,2)) # last action matrix
        R = [0, 0]     # last choice structure
        LL = 0 # log likelihood

        for i in range(len(self.data["stimuli"])): 
            
            if self.data["stimuli"][i] == 2 or self.data["stimuli"][i] == 4:
                R = R = np.flipud(R) 

            s1 = self.data["s"][i][0] # first choice stage
            s2 = self.data["s"][i][1] # second choice stage
            action = self.data["choice"][i]
            a = action[0] - (s1 == 2) * 2

            # choose which weight to update based on stake of trial
            if self.data["stake"][i] == 1:
                w = w_lo
            else: 
                w = w_hi

            # calculating the Q values for model-based learning
            Q_MB = TM[s1-1].T @ Q2

            # calculating the Q values for mixing model
            Q = w*Q_MB + (1-w)*Q_MF[s1-1, :].T + st * M[s1-1, :].T + respst * np.array(R)

            # calculating the probability choosing action a given Q
            LL = LL + b*Q[a-1] - logsumexp(b * Q) 

            M = np.zeros((2,2))
            M[s1-1, a-1] = 1

            R = np.zeros((2,1))
            if action == self.data["stimuli"][i][0]:
                R[0] = 1
            else: 
                R[1] = 1

            # update Q values for model-free learning
            # first choice stage
            dtQ1 = Q2[s2-1] - Q_MF[s1-1, a-1]
            Q_MF[s1-1, a-1] = Q_MF[s1-1, a-1] + lr * dtQ1
            # second choice stage
            dtQ2 = self.data["points"][i] - Q2[s2-1]
            Q2[s2-1] = Q2[s2-1] + lr * dtQ2
            Q_MF[s1-1, a-1] = Q_MF[s1-1, a-1] + lamb * lr * dtQ2

        # compile results
        results = pd.DataFrame({
            "userID": self.data["userID"][0],
            "LL": [LL],
            "Q": [Q.flatten()],
            "Q_MF": [Q_MF.flatten()],
            "Q_MB": [Q_MB.flatten()]
        })

        return results    