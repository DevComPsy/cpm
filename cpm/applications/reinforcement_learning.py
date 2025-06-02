from cpm.generators import Wrapper, Parameters, Value

import cpm
import numpy
import pandas
import warnings
import ipyparallel as ipp  ## for parallel computing with ipython (specific for Jupyter Notebook)


from scipy.special import logsumexp

class RLRW(Wrapper):
    """
    The class implements a simple reinforcement learning model for a multi-armed bandit tasks using a standard update rule calculating prediction error and a Softmax decision rule.
    The model is an n-dimensional and k-armed implementation of model 3 from Wilson and Collins (2019).

    Parameters
    ----------
    data: pandas.DataFrame
        The data to be fit by the model. The data must contain columns for the choice and reward for each dimension. See Notes for more information on what columns should you include.
    dimensions: int
        The number of distinct stimuli present in the data.
    parameters_settings: list-like
        The parameters to be fit by the model. The parameters must be specified as a list of lists, with each list containing the value, lower, and upper bounds of the parameter. See Notes for more information on how to specify parameters and for the default settings.

    Returns
    -------
    cpm.generators.Wrapper
        A cpm.generators.Wrapper object.

    Examples
    --------
    >>> import numpy
    >>> import pandas
    >>> from cpm.applications import RLRW
    >>> from cpm.datasets import load_bandit_data

    >>> twoarm = load_bandit_data()
    >>> model = RLRW(data=data, dimensions=4)
    >>> model.run()


    Notes
    -----
    Data must contain the following columns:

    - choice: the choice of the participant from the available options, starting from 0.
    - arm_n: the stimulus identifier for each option (arms in the bandit task), where n is the option available on a given trial. If there are more than one options, the stimulus identifier should be specified as separate columns of arm_1, arm_2, arm_3, etc. or arm_left, arm_middle, arm_right, etc.
    - reward_n: the reward given after each options, where n is the corresponding arm of the bandit available on a given trial. If there are more than one options, the reward should be specified as separate columns of reward_1, reward_2, reward_3, etc.

    parameters_settings must be a 2D array, like [[0.5, 0, 1], [5, 1, 10]], where the first list specifies the alpha parameter and the second list specifies the temperature parameter. The first element of each list is the initial value of the parameter, the second element is the lower bound, and the third element is the upper bound. The default settings are 0.5 for alpha with a lower bound of 0 and an upper bound of 1, and 5 for temperature with a lower bound of 1 and an upper bound of 10.

    References
    ----------
    Robert C Wilson Anne GE Collins (2019) Ten simple rules for the computational modeling of behavioral data eLife 8:e49547.

    """

    def __init__(
        self, data=None, dimensions=2, parameters_settings=None, generate=False
    ):
        if parameters_settings is None:
            parameters_settings = [[0.5, 0, 1], [5, 0, 10]]
            warnings.warn("No parameters specified, using default parameters.")
        parameters = Parameters(
            # freely varying parameters are indicated by specifying priors
            alpha=Value(
                value=parameters_settings[0][0],
                lower=parameters_settings[0][1],
                upper=parameters_settings[0][2],
                prior="truncated_normal",
                args={"mean": 0.5, "sd": 0.25},
            ),
            temperature=Value(
                value=parameters_settings[1][0],
                lower=parameters_settings[1][1],
                upper=parameters_settings[1][2],
                prior="truncated_normal",
                args={"mean": 5, "sd": 2.5},
            ),
            values=numpy.ones(dimensions) / dimensions,
        )

        @ipp.require("numpy")
        def model(parameters, trial, generate=generate):
            # pull out the parameters
            alpha = parameters.alpha
            temperature = parameters.temperature
            values = numpy.array(parameters.values)
            ## first we get the bandits and their corresponding stimulus identifier
            arm_names = [
                col for col in trial.index if "arm" in col
            ]  ## get column names beginning with stimulus
            arms = numpy.array(
                [trial[i] for i in arm_names]
            )  ## stimulus identifier for each arm of the bandit
            k_arms = arms.shape[0]  ## number of arms
            dims = values.shape[0]  ## number of stimuli
            choice = trial.response.astype(int)
            reward_names = [
                col for col in trial.index if "reward" in col
            ]  ## get column names beginning with stimulus
            feedback = numpy.array(
                [trial[i] for i in reward_names]
            )  ## compile reward vector
            ## get the activations for each arm given q-values for each stimulus
            activations = numpy.array([values[i - 1] for i in arms])
            activations = activations.reshape(k_arms, 1)
            ## compute softmax
            response = cpm.models.decision.Softmax(
                activations=activations, temperature=temperature
            )
            response.compute()
            ## check for NaN in policy
            if numpy.isnan(response.policies).any():
                # if the policy is NaN for a given action, then we need to set it to 1 to avoid numerical issues
                warnings.warn(
                    f"NaN in policy with parameters: {alpha.value}, {temperature.value}, \nand with policy: {response.policies}\n"
                )
                response.policies[numpy.isnan(response.policies)] = 1
            # if generate is true, generate a response from softmax probabilities
            if generate:
                choice = response.choice()
            ## match choice to stimulus identifier
            stim_choice = arms[choice] - 1
            # update the values for that stimulus
            mute = numpy.zeros(dims)
            mute[stim_choice] = (
                1  ## determine which stimulus' q-values we need to update
            )
            teacher = feedback[choice]  ## get reward for that bandit
            update = cpm.models.learning.SeparableRule(
                weights=values, feedback=[teacher], input=mute, alpha=alpha
            )
            update.compute()

            values += update.weights.flatten()
            ## compile output
            output = {
                "policy": response.policies,  # policies
                "reward": teacher,  # reward of the chosen action
                "values": values,  # updated values
                "change": update.weights,  # change in the values - prediction error
                "dependent": numpy.array(
                    [response.policies[1]]
                ),  # dependent variable P(choosing the right | stimuli on right)
            }
            return output

        super().__init__(data=data, model=model, parameters=parameters)


class MBMF:
    """
    Model for model-based vs model-free decision making in a decision making task

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

    def __init__(
        self, data=None, dimensions=2, parameters_settings=None, generate=False
    ):
        self.data = data

        ## TODO: cpm.generators.Value should be used here
        ## TODO: cpm.generators.Parameters should be used here
        if parameters is None:
            self.parameters = {
                "temperature": 0.8,
                "learning_rate": 0.5,
                "eligibility_trace_decay": 0.8,
                "mixing_weight_low_stake": 0.8,
                "mixing_weight_high_stake": 0.8,
                "stickiness": -0.1,
                "response_stickiness": -0.1
            }
            warnings.warn("No parameters specified, using default parameters.")
        else:
            self.parameters = {
                "temperature": parameters[0],
                "learning_rate": parameters[1],
                "eligibility_trace_decay": parameters[2],
                "mixing_weight_low_stake": parameters[3],
                "mixing_weight_high_stake": parameters[4],
                "stickiness": parameters[5],
                "response_stickiness": parameters[6]
            }


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
        b = self.parameters["temperature"] # softmax inverse temperature
        lr = self.parameters["learning_rate"] # learning rate
        lamb = self.parameters["eligibility_trace_decay"] # eligibility trace decay
        w_lo = self.parameters["mixing_weight_low_stake"] # mixing weight for low stake trials
        w_hi = self.parameters["mixing_weight_high_stake"] # mixing weight for high stake trials
        st = self.parameters["stickiness"] # stickiness
        respst = self.parameters["response_stickiness"] # response stickiness

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

            # update choice structure
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

    super().__init__(data=data, model=model, parameters=parameters)
