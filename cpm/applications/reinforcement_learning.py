from cpm.generators import Wrapper, Parameters, Value

import cpm
import numpy
import pandas
import warnings
import ipyparallel as ipp  ## for parallel computing with ipython (specific for Jupyter Notebook)


class RLRW(Wrapper):
    r"""
    The class implements a simple reinforcement learning model for a multi-armed bandit tasks using a standard update rule calculating prediction error and a Softmax decision rule.
    The model is an n-dimensional and k-armed implementation of model 3 from Wilson and Collins (2019), which largely corresponds to the model presented by Sutton & Barto (2020) in Chapter 14.

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
    >>> from cpm.applications.reinforcement_learning import RLRW
    >>> from cpm.datasets import load_bandit_data
    >>> data = load_bandit_data()
    >>> model = RLRW(data=data[data.ppt == 1], dimensions=4)
    >>> model.run()
    >>> model.export().head()


    Notes
    -----

    The model implementation uses two parameters:

    - alpha: the learning rate, which determines how much the model updates its values based on the prediction error.
    - temperature: the inverse temperature, which determines the choice stochasticity -- how sensitive is the model to value differences.

    Data must contain the following columns:

    - response: the choice of the participant from the available options, starting from 0.
    - arm_n: the stimulus identifier for each option (arms in the bandit task), where n is the option available on a given trial. If there are more than one options, the stimulus identifier should be specified as separate columns of arm_1, arm_2, arm_3, etc. or arm_left, arm_middle, arm_right, etc.
    - reward_n: the reward given after each options, where n is the corresponding arm of the bandit available on a given trial. If there are more than one options, the reward should be specified as separate columns of reward_1, reward_2, reward_3, etc.

    parameters_settings must be a 2D array, like [[0.5, 0, 1], [5, 0, 10]], where the first list specifies the alpha parameter and the second list specifies the temperature parameter. The first element of each list is the initial value of the parameter, the second element is the lower bound, and the third element is the upper bound. The default settings are 0.5 for alpha with a lower bound of 0 and an upper bound of 1, and 5 for temperature with a lower bound of 0 and an upper bound of 10.

    The model is defined as follows:

    Let each stimulus have an associated value, which is the expected reward that can be obtained from selecting that stimulus. Let also :math:`Q(a)` be the estimated value of action :math:`a`. In each trial, :math:`t`, there are two stimuli present, so :math:`Q(a)` could be :math:`Q(\text{left})` or :math:`Q(\text{right})`, where the corresponding :math:`Q` values are derived from the associated value of the stimulus present on the left or right. In each trial :math:`t` , the Softmax choice rule (Bridle, 1990) will convert the estimated value of each action into a probability according to the following policy:

    .. math::

        P(a) = \frac{e^{Q(a)/\tau}}{\sum_{a'} e^{Q(a')/\tau}}

    where :math:`\tau` is the temperature parameter. After the choice is made and feedback is received, the value of the chosen stimulus is updated. The current implementation uses the variant of the delta rule (Rescorla & Wagner, 1972; Rumelhart, Hinton, & Williams, 1986) adapted for multi-armed bandit problems where each option has a single stimulus dimension (Sutton & Barto, 2020), reducing Rescorla-Wagner's summed error-term to the following equation, similar to (Bush & Mosteller, 1951):

    .. math::

        \Delta Q(a_t) = \alpha \times [R - Q(a_t)]

    where :math:`\alpha` is the learning rate, :math:`R` is the reward received for the chosen action, and :math:`Q(a_t)` is the estimated value of the chosen action before updating. The values of unchosen stimuli remain unchanged.

    References
    ----------

    Bridle, J. S. (1990). Probabilistic Interpretation of Feedforward Classification Network Outputs, with Relationships to Statistical Pattern Recognition. In F. F. Soulié & J. Hérault (Eds.), Neurocomputing (pp. 227–236). Springer. https://doi.org/10.1007/978-3-642-76153-9_28

    Bush, R. R., & Mosteller, F. (1951). A mathematical model for simple learning. Psychological Review, 58(5), 313–323. https://doi.org/10.1037/h0054388

    Rescorla, R. A. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and non-reinforcement. Classical Conditioning, Current Research and Theory, 2, 64–69.

    Robert C Wilson & Anne GE Collins (2019) Ten simple rules for the computational modeling of behavioral data eLife 8:e49547.

    Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533–536. https://doi.org/10.1038/323533a0

    Sutton, R. S., & Barto, A. (2020). Reinforcement learning: An introduction (Second edition). The MIT Press.

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
            values = numpy.asarray(parameters.values).copy()
            ## first we get the bandits and their corresponding stimulus identifier
            arm_names = [
                col for col in trial.index if "arm" in col
            ]  ## get column names beginning with stimulus
            arms = numpy.array(
                [trial[i] for i in arm_names],
                dtype=int,
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
                "values": values.copy(),  # updated values
                "change": update.weights,  # change in the values - prediction error
                "dependent": numpy.asarray(
                    [response.policies[1]]
                ),  # dependent variable P(choosing the right | stimuli on right)
            }
            return output

        super().__init__(data=data, model=model, parameters=parameters)


class HybridMBMF(Wrapper):
    r"""
    The class implements the hybrid model-based / model-free reinforcement learning model for the deterministic two-step task (Kool et al., 2016), in the 6-parameter variant used by Smid et al. (2022).
    Model-free values are learned with a SARSA rule with an eligibility trace, model-based values are computed from the known transition structure, and the two are mixed before a Softmax decision rule.

    Parameters
    ----------
    data: pandas.DataFrame
        The data to be fit by the model. See Notes for the columns required.
    parameters_settings: list-like
        The parameters to be fit by the model. The parameters must be specified as a list of lists, with each list containing the value, lower, and upper bounds of the parameter. See Notes for more information on how to specify parameters and for the default settings.
    q_init: float
        The initial value of all model-free and second-stage values. It should be on the same scale as the rewards, for example, 0.5 for rewards scaled between 0 and 1, or 4.5 for rewards between 0 and 9. Default is 0.5.
    generate: bool
        If True, the model samples a first-stage action on each trial from its policy instead of using the observed one, and the reward is taken from the reward schedule of the reached second-stage state. Default is False.

    Returns
    -------
    cpm.generators.Wrapper
        A cpm.generators.Wrapper object.

    Examples
    --------
    >>> import numpy
    >>> import pandas
    >>> from cpm.applications.reinforcement_learning import HybridMBMF

    >>> data = pandas.DataFrame({
    ...     "s1": [0, 1, 0],
    ...     "action": [1, 0, 0],
    ...     "s2": [0, 1, 1],
    ...     "reward": [0.7, 0.2, 0.4],
    ...     "stimuli_first": [0, 1, 0],
    ...     "observed": [1, 0, 0],
    ... })
    >>> model = HybridMBMF(data=data, parameters_settings=[
    ...     [2, 0, 5], [0.5, 0, 1], [0.5, 0, 1], [0.5, 0, 1], [0, -5, 5], [0, -5, 5]
    ... ])
    >>> model.run()
    >>> model.dependent.shape
    (3, 1)

    Notes
    -----

    The task has two first-stage starting states, each offering two actions. Transitions are deterministic and shared across starting states: action 0 leads to second-stage state 1 and action 1 leads to second-stage state 0. The second-stage states pay out a drifting reward, and there is no second-stage choice.

    The model implementation uses six parameters:

    - inv_temperature: the Softmax inverse temperature, :math:`\beta`.
    - learning_rate: the learning rate for the SARSA updates, :math:`\alpha`.
    - eligibility_trace: the eligibility trace decay, :math:`\lambda`.
    - mb_weight: the model-based mixing weight, :math:`w`, where 0 is purely model-free and 1 is purely model-based.
    - choice_stickiness: the bonus for repeating the last action taken in the same starting state, :math:`\pi`.
    - response_stickiness: the bonus for repeating the last screen position (left or right), :math:`\rho`.

    parameters_settings must be a 2D array, like [[2, 0, 5], [0.5, 0, 1], [0.5, 0, 1], [0.5, 0, 1], [0, -5, 5], [0, -5, 5]], where each list specifies one parameter in the order above. The first element of each list is the initial value of the parameter, the second element is the lower bound, and the third element is the upper bound. The default settings are the ones in the example.

    Data must contain the following columns:

    - s1: the first-stage starting state, 0 or 1.
    - stimuli_first: the action displayed on the left side of the screen, 0 or 1. Optional; if absent, action 0 is assumed to be on the left.
    - action: the chosen first-stage action, 0 or 1. Not required if `generate` is True.
    - s2: the second-stage state reached, 0 or 1. Not required if `generate` is True.
    - reward: the reward received in the second-stage state. Not required if `generate` is True.
    - position: the screen position of the chosen action, 0 for left and 1 for right. Optional; if absent, it is derived from `stimuli_first` and `action`.
    - reward_0, reward_1: the reward each second-stage state would pay out on the trial. Only required if `generate` is True.
    - observed: the chosen first-stage action, 0 or 1, used by the loss function. The dependent variable of the model is the probability of choosing action 1.

    The model is defined as follows (Kool et al., 2016; Smid et al., 2022). The model-based value of each first-stage action is computed from the transition matrix, :math:`T`, and the second-stage values, :math:`Q_2`:

    .. math::

        Q_{MB}(a) = \sum_{s_2} T(a, s_2) Q_2(s_2)

    The model-based and model-free values are combined with the stickiness terms into a single hybrid value:

    .. math::

        Q_{hybrid}(s_1, a) = w Q_{MB}(a) + (1 - w) Q_{MF}(s_1, a) + \pi M(s_1, a) + \rho R(a)

    where :math:`M(s_1, a)` is 1 if :math:`a` was chosen on the previous trial and the previous starting state was also :math:`s_1`, and 0 otherwise, and :math:`R(a)` is 1 if :math:`a` is displayed at the same screen position as the response on the previous trial, and 0 otherwise. The hybrid values are converted into a policy with the Softmax choice rule (Bridle, 1990):

    .. math::

        P(a) = \frac{e^{\beta Q_{hybrid}(s_1, a)}}{\sum_{a'} e^{\beta Q_{hybrid}(s_1, a')}}

    After the choice, the model-free and second-stage values are updated with a SARSA rule with an eligibility trace, see :class:`cpm.models.learning.SARSATrace`.

    References
    ----------

    Bridle, J. S. (1990). Probabilistic Interpretation of Feedforward Classification Network Outputs, with Relationships to Statistical Pattern Recognition. In F. F. Soulié & J. Hérault (Eds.), Neurocomputing (pp. 227–236). Springer. https://doi.org/10.1007/978-3-642-76153-9_28

    Kool, W., Cushman, F. A., & Gershman, S. J. (2016). When does model-based control pay off? PLoS Computational Biology, 12(8), e1005090. https://doi.org/10.1371/journal.pcbi.1005090

    Smid et al. (2022). Computational and behavioral correlates of developmental changes in model-based/model-free decision-making. Developmental Science, e13380.

    """

    def __init__(self, data=None, parameters_settings=None, q_init=0.5, generate=False):
        if parameters_settings is None:
            parameters_settings = [
                [2, 0, 5],
                [0.5, 0, 1],
                [0.5, 0, 1],
                [0.5, 0, 1],
                [0, -5, 5],
                [0, -5, 5],
            ]
            warnings.warn("No parameters specified, using default parameters.")
        priors = [
            {"mean": 1.5, "sd": 2.0},
            {"mean": 0.5, "sd": 0.25},
            {"mean": 0.5, "sd": 0.25},
            {"mean": 0.5, "sd": 0.25},
            {"mean": 0.0, "sd": 1.0},
            {"mean": 0.0, "sd": 1.0},
        ]
        names = [
            "inv_temperature",
            "learning_rate",
            "eligibility_trace",
            "mb_weight",
            "choice_stickiness",
            "response_stickiness",
        ]
        parameters = Parameters(
            # freely varying parameters are indicated by specifying priors
            **{
                name: Value(
                    value=settings[0],
                    lower=settings[1],
                    upper=settings[2],
                    prior="truncated_normal",
                    args=prior,
                )
                for name, settings, prior in zip(names, parameters_settings, priors)
            },
            # internal states of the model
            q_mf=numpy.full((2, 2), q_init, dtype=float),
            q2=numpy.full(2, q_init, dtype=float),
            m=numpy.zeros((2, 2)),
            r=numpy.zeros(2),
        )
        ## deterministic transitions: action 0 -> state 1, action 1 -> state 0
        transitions = numpy.array([[0.0, 1.0], [1.0, 0.0]])

        @ipp.require("numpy")
        def model(parameters, trial, generate=generate):
            # pull out the parameters
            inv_temperature = parameters.inv_temperature
            learning_rate = parameters.learning_rate
            eligibility_trace = parameters.eligibility_trace
            mb_weight = parameters.mb_weight
            choice_stickiness = parameters.choice_stickiness
            response_stickiness = parameters.response_stickiness
            q_mf = numpy.asarray(parameters.q_mf).copy()
            q2 = numpy.asarray(parameters.q2).copy()
            m = numpy.asarray(parameters.m).copy()
            r = numpy.asarray(parameters.r).copy()

            s1 = int(trial.s1)
            stimuli_first = int(trial.get("stimuli_first", 0))
            ## response stickiness is stored by screen position [left, right],
            ## so flip it to action space when action 1 is displayed on the left
            r_action = r[::-1] if stimuli_first == 1 else r

            ## hybrid values
            q_mb = transitions @ q2
            q_hybrid = (
                mb_weight * q_mb
                + (1 - mb_weight) * q_mf[s1]
                + choice_stickiness * m[s1]
                + response_stickiness * r_action
            )
            response = cpm.models.decision.Softmax(
                activations=q_hybrid, temperature=inv_temperature
            )
            response.compute()

            if generate:
                action = response.choice()
                s2 = int(numpy.argmax(transitions[action]))
                reward = float(trial[f"reward_{s2}"])
                position = stimuli_first ^ action
            else:
                action = int(trial.action)
                s2 = int(trial.s2)
                reward = float(trial.reward)
                position = int(trial.get("position", stimuli_first ^ action))

            ## update stickiness for the next trial
            m = numpy.zeros((2, 2))
            m[s1, action] = 1
            r = numpy.zeros(2)
            r[position] = 1

            ## update values
            update = cpm.models.learning.SARSATrace(
                learning_rate=learning_rate,
                eligibility_trace=eligibility_trace,
                model_free_values=q_mf,
                second_stage_values=q2,
                starting_state=s1,
                action=action,
                reached_second_stage=s2,
                reward=reward,
            )
            model_free_delta, planet_value_delta = update.compute()
            q_mf += model_free_delta
            q2 += planet_value_delta

            ## compile output
            output = {
                "policy": response.policies,  # policies
                "action": action,  # chosen (or generated) first-stage action
                "s2": s2,  # second-stage state reached
                "reward": reward,  # reward received
                "position": position,  # screen position of the response
                "stage1_prediction_error": update.stage1_prediction_error,
                "stage2_prediction_error": update.stage2_prediction_error,
                "q_mf": q_mf.copy(),  # updated model-free values
                "q2": q2.copy(),  # updated second-stage values
                "m": m,  # choice stickiness for the next trial
                "r": r,  # response stickiness for the next trial
                "dependent": numpy.asarray(
                    [response.policies[1]]
                ),  # dependent variable P(choosing action 1)
            }
            return output

        super().__init__(data=data, model=model, parameters=parameters)
