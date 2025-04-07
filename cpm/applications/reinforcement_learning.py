from cpm.generators import Wrapper, Parameters, Value

import cpm
import numpy
import pandas
import warnings
import ipyparallel as ipp  ## for parallel computing with ipython (specific for Jupyter Notebook)


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
