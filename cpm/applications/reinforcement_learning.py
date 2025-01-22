from cpm.generators import Wrapper, Parameters, Value

import cpm
import numpy
import pandas
import warnings
import ipyparallel as ipp  ## for parallel computing with ipython (specific for Jupyter Notebook)


class RLRW(Wrapper):
    """
    The class implements a simple reinforcement learning model for a multi-armed bandit task using a standard update rule calculating prediction error and a Softmax decision rule.
    The model is an n-dimensional implementation of model 3 from Wilson and Collins (2019).

    Parameters
    ----------
    data: pandas.DataFrame
        The data to be fit by the model. The data must contain columns for the choice and reward for each dimension. See Notes for more information on what columns should you include.
    dimensions: int
        The number of distinct stimuli present in the data.
    parameters_settings: list-like
        The parameters to be fit by the model. The parameters must be specified as a list of lists, with each list containing the value, lower, and upper bounds of the parameter. See Notes for more information on how to specify parameters and for the default settings.

    Notes
    -----
    Data must contain the following columns:

    - choice: the choice of the participant from the available options, starting from 0.
    - reward_n: the reward given after each options, where n is the option available on a given trial. If there are more than one options, the reward should be specified as separate columns of reward_1, reward_2, reward_3, etc.

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
            values = numpy.array(
                parameters.values
            )  # copy essentially prevents us from accidentally overwriting the original values
            dims = values.shape[0]
            choice = trial.choice.astype(int)
            feedback = numpy.array(
                [trial[f"reward_{i}"] for i in numpy.arange(1, dims + 1)]
            )  ## compile reward vector

            mute = numpy.zeros(dims)
            activations = values.reshape(dims, 1)
            response = cpm.models.decision.Softmax(
                activations=activations, temperature=temperature
            )
            response.compute()

            if numpy.isnan(response.policies).any():
                # if the policy is NaN for a given action, then we need to set it to 1
                print(
                    f"NaN in policy with parameters: {alpha.value}, {temperature.value}\n"
                )
                print(response.policies)
                response.policies[numpy.isnan(response.policies)] = 1
            # if generate is true, generate a response from softmax probabilities
            if generate:
                choice = response.choice()
            # \ update the values
            mute = numpy.zeros(dims)
            mute[choice] = 1  ## determine the
            teacher = feedback[choice]
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
                "dependent": numpy.array([response.policies[1]]),  # dependent variable
            }
            return output

        super().__init__(data=data, model=model, parameters=parameters)
