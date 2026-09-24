import numpy as np

__all__ = [
    "DeltaRule",
    "SeparableRule",
    "QLearningRule",
    "HumbleTeacher",
    "SARSATrace",
]


class DeltaRule:
    """
    DeltaRule class computes the prediction error for a given input and target value.

    Parameters
    ----------
    alpha : float
        The learning rate.
    zeta  : float
        The constant fraction of the magnitude of the prediction error.
    weights : array-like
        The value matrix, where rows are outcomes and columns are stimuli or features. The values can be anything; for example belief values, association weights, connection weights, Q-values.
    feedback : array-like
        The target values or feedback, sometimes referred to as teaching signals. These are the values that the algorithm should learn to predict.
    input : array-like
        The input value. The stimulus representation in the form of a 1D array, where each element can take a value of 0 and 1.
    **kwargs : dict, optional
        Additional keyword arguments.

    See Also
    --------
    [cpm.models.learning.SeparableRule][cpm.models.learning.SeparableRule] : A class representing a learning rule based on the separable error-term of Bush and Mosteller (1951).

    Notes
    -----

    The delta-rule is a summed error term, which means that the error is defined as
    the difference between the target value and the summed activation of all values
    for a given output units target value available on the current trial/state. For separable
    error term, see the Bush and Mosteller (1951) rule.

    The current implementation is based on the Gluck and Bower's (1988) delta rule, an
    extension of the Rescorla and Wagner (1972) learning rule to multi-outcome learning. Such that


    $$
    \\Delta w_{ij} = \\alpha \\cdot (\\lambda_i - \\sum_j w_{ij}) \\cdot x_j
    $$

    where $\\Delta w_{ij}$ is the change in weight for the $j$-th stimulus for the $i$-th outcome,
    $\\lambda_i$ is the target (feedback) value for the i-th outcome, $w_ij$ is the weights of stimulus $j$
    for the $i$-th outcome,
    $x_j$ is the j-th stimulus input, and $\\alpha$ is the learning rate. This is consistent with the
    Rescorla and Wagner (1972)'s learning rule incorporating the summed error term.

    Examples
    --------
    >>> import numpy as np
    >>> from cpm.models.learning import DeltaRule
    >>> weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    >>> teacher = np.array([1, 0])
    >>> input = np.array([1, 1, 0])
    >>> delta_rule = DeltaRule(alpha=0.1, zeta=0.1, weights=weights, feedback=teacher, input=input)
    >>> delta_rule.compute()
    array([[ 0.07,  0.07,  0.  ],
           [-0.09, -0.09, -0.  ]])
    >>> delta_rule.noisy_learning_rule()
    array([[ 0.05755793,  0.09214091,  0.],
           [-0.08837513, -0.1304325 ,  0.]])

    This implementation generalises to n-dimensional matrices, which means
    that it can be applied to both single- and multi-outcome learning paradigms.

    >>> weights = np.array([0.1, 0.6, 0., 0.3])
    >>> teacher = np.array([1])
    >>> input = np.array([1, 1, 0, 0])
    >>> delta_rule = DeltaRule(alpha=0.1, weights=weights, feedback=teacher, input=input)
    >>> delta_rule.compute()
    array([[0.03, 0.03, 0.  , 0.  ]])

    References
    ---------
    Gluck, M. A., & Bower, G. H. (1988). From conditioning to category learning: An adaptive network model. Journal of Experimental Psychology: General, 117(3), 227–247.

    Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement. In A. H. Black & W. F. Prokasy (Eds.), Classical conditioning II: Current research and theory (pp. 64-99). New York:Appleton-Century-Crofts.

    Widrow, B., & Hoff, M. E. (1960, August). Adaptive switching circuits. In IRE WESCON convention record (Vol. 4, No. 1, pp. 96-104).
    """

    def __init__(
        self,
        alpha=None,
        zeta=None,
        weights=None,
        feedback=None,
        input=None,
        **kwargs,
    ):
        self.alpha = alpha
        self.zeta = zeta

        self.weights = [[]]
        if weights is not None:
            self.weights = np.asarray(weights.copy())
        self.teacher = feedback
        self.input = np.asarray(input)
        self.shape = self.weights.shape
        if len(self.shape) == 1:
            self.shape = (1, self.shape[0])
            self.weights = np.array([self.weights])
        self.error = np.zeros(self.shape[0])
        self.__run__ = False

    def compute(self):
        """
        Compute the prediction error using the delta learning rule. It is based on the
        Gluck and Bower's (1988) delta rule, an extension to Rescorla and Wagner
        (1972), which was identical to that of Widrow and Hoff (1960).

        Returns
        -------
        ndarray
            The prediction error for each stimuli-outcome mapping with learning noise.
            It has the same shape as the weights input argument.
        """

        for i in range(self.shape[0]):
            # calculate summed error for a given output unit
            activations = np.sum(self.weights[i] * self.input)
            self.error[i] = self.teacher[i] - activations
            for j in range(self.shape[1]):
                # calcualte the change on weights
                self.weights[i, j] = self.alpha * self.error[i] * self.input[j]
        self.__run__ = True
        return self.weights

    def noisy_learning_rule(self):
        """
        Add random noise to the prediction error computed from the delta learning rule as specified
        Findling et al. (2019). It is inspired by Weber's law of intensity
        sensation.

        Returns
        -------
        ndarray
            The prediction error for each stimuli-outcome mapping with learning noise.
            It has the same shape as the weights input argument.

        References
        ----------

        Findling, C., Skvortsova, V., Dromnelle, R., Palminteri, S., and Wyart, V. (2019). Computational noise in reward-guided learning drives behavioral variability in volatile environments. Nature Neuroscience 22, 2066–2077
        """
        if not self.__run__:
            self.compute()
        # random noise vector initialized with zeros
        epsilon = np.zeros_like(self.error)
        for i in range(self.shape[0]):
            # calculate standard deviation of the noise
            sigma = self.zeta * np.abs(self.error[i])
            # select random noise from normal distribution
            epsilon[i] = np.random.normal(0, sigma)
            # add noise to the weight changes for stimuli present on trial
            self.weights[i] = self.weights[i] + epsilon[i] * self.input
        return self.weights

    def reset(self):
        """
        Reset the weights to zero.
        """
        self.weights = np.zeros(self.shape)

    def __repr__(self):
        return f"DeltaRule(alpha={self.alpha},\n weights={self.weights},\n teacher={self.teacher})"

    def __str__(self):
        return f"DeltaRule(alpha={self.alpha},\n weights={self.weights},\n teacher={self.teacher})"

    def __call__(self):
        return self.compute()


class SeparableRule:
    """
    A class representing a learning rule based on the separable error-term of
    Bush and Mosteller (1951).

    Parameters
    -----------
    alpha : float
        The learning rate.
    zeta : float, optional
        The constant fraction of the magnitude of the prediction error, also called Weber's scaling.
    weights : array-like
        The value matrix, where rows are outcomes and columns are stimuli or features. The values can be anything; for example belief values, association weights, connection weights, Q-values.
    feedback : array-like, optional
        The target values or feedback, sometimes referred to as teaching signals. These are the values that the algorithm should learn to predict.
    input : array-like, optional
        The input value. The stimulus representation in the form of a 1D array, where each element can take a value of 0 and 1.
    **kwargs : dict, optional
        Additional keyword arguments.

    See Also
    --------
    [cpm.models.learning.DeltaRule][cpm.models.learning.DeltaRule] : An extension of the Rescorla and Wagner (1972) learning rule by Gluck and Bower (1988) to allow multi-outcome learning.

    Notes
    -----
    This type of learning rule was among the earliest formal models of associative learning (Le Pelley, 2004), which were based on standard linear operators (Bush & Mosteller, 1951; Estes, 1950; Kendler, 1971). It is used in a variety of reinforcement learning models. This learning rule is defined in `cpm` as


    $$
    \\Delta w_{ij} = \\alpha \\cdot (\\lambda_i - w_{ij}) \\cdot x_j
    $$

    which is consistent with the modification of the Rescorla and Wagner (1972) learning rule by Sutton and Barto (2018). The current implementation generalises to any number of outcomes and stimuli, which means that it can be applied to both single- and multi-outcome learning paradigms.

    References
    ----------
    Bush, R. R., & Mosteller, F. (1951). A mathematical model for simple learning. Psychological Review, 58, 313–323

    Estes, W. K. (1950). Toward a statistical theory of learning. Psychological Review, 57, 94–107

    Kendler, T. S. (1971). Continuity theory and cue dominance. In J. T. Spence (Ed.), Essays in neobehaviorism: A memorial volume to Kenneth W. Spence. New York: Appleton-Century-Crofts.

    Le Pelley, M. E. (2004). The role of associative history in models of associative learning: A selective review and a hybrid model. Quarterly Journal of Experimental Psychology Section B, 57(3), 193-243.

    """

    def __init__(
        self, alpha=None, zeta=None, weights=None, feedback=None, input=None, **kwargs
    ):
        self.alpha = alpha
        self.zeta = zeta

        self.weights = [[]]
        if weights is not None:
            self.weights = weights.copy()
        self.teacher = feedback
        self.input = np.asarray(input)
        self.shape = self.weights.shape
        if len(self.shape) == 1:
            self.shape = (1, self.shape[0])
            self.weights = np.array([self.weights])
        self.error = np.zeros(self.shape)
        self.__run__ = False

    def compute(self):
        """
        Computes the prediction error using the learning rule.

        Returns:
        --------
        ndarray
            The prediction error for each stimuli-outcome mapping.
            It has the same shape as the weights input argument.

        Notes
        -----
        The prediction error for each stimuli-outcome mapping before the update is stored in `error`,
        which has the same shape as the weights.
        """
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # separable prediction error for each outcome-stimulus pair
                self.error[i, j] = self.teacher[i] - self.weights[i, j]
                self.weights[i, j] = self.alpha * self.error[i, j] * self.input[j]
        self.__run__ = True
        return self.weights

    def noisy_learning_rule(self):
        """
        Add random noise to the prediction error computed from the delta learning rule as specified
        Findling et al. (2019). It is inspired by Weber's law of intensity
        sensation.

        Returns
        -------
        ndarray
            The prediction error for each stimuli-outcome mapping with learning noise.
            It has the same shape as the weights input argument.

        References
        ----------

        Findling, C., Skvortsova, V., Dromnelle, R., Palminteri, S., and Wyart, V. (2019). Computational noise in reward-guided learning drives behavioral variability in volatile environments. Nature Neuroscience 22, 2066–2077
        """
        if not self.__run__:
            self.compute()
        epsilon = np.zeros_like(self.error)
        for i in range(self.shape[0]):
            sigma = self.zeta * np.abs(self.error[i])
            epsilon[i] = np.random.normal(0, sigma)
            self.weights[i] = self.weights[i] + epsilon[i] * self.input
        return self.weights

    def reset(self):
        """
        Resets the weights to zero.
        """
        self.weights = np.zeros(self.shape)

    def __repr__(self):
        return f"SeparableRule(alpha={self.alpha},\n weights={self.weights},\n teacher={self.teacher})"

    def __str__(self):
        return f"SeparableRule(alpha={self.alpha},\n weights={self.weights},\n teacher={self.teacher})"

    def __call__(self):
        return self.compute()


class QLearningRule:
    """
    Q-learning rule (Watkins, 1989) for a one-dimensional array of Q-values.

    Parameters
    ----------
    alpha : float
        The learning rate. Default is 0.5.
    gamma : float
        The discount factor. Default is 0.1.
    values : ndarray
        The values matrix.  It is a 1D array of Q-values active for the current state, where each element corresponds to an action.
    reward : float
        The reward received on the current state.
    maximum : float
        The maximum estimated reward for the next state.

    Notes
    -----
    The Q-learning rule is a model-free reinforcement learning algorithm that is used to learn the value of an action in a given state.
    It is defined as

    $$
    \\Delta \\mathcal{Q}(s, a) =  \\alpha \\cdot (r + \\gamma \\cdot \\max_{a'} \\mathcal{Q}(s', a') - \\mathcal{Q}(s, a))
    $$

    where $\\Delta \\mathcal{Q}(s, a)$ is the change in value of action $a$ in state $s$, $r$ is the reward received on the current state, $\\gamma$ is the discount factor, and $\\max_{a'} \\mathcal{Q}(s', a')$ is the maximum estimated reward for the next state.

    Examples
    --------
    >>> import numpy as np
    >>> from cpm.models.learning import QLearningRule
    >>> values = np.array([1, 0.5, 0.99])
    >>> component = QLearningRule(alpha=0.1, gamma=0.8, values=values, reward=1, maximum=10)
    >>> component.compute()
    array([1.8  , 1.35 , 1.791])

    References
    ----------
    Watkins, C. J. C. H. (1989). Learning from delayed rewards.

    Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8, 279-292.
    """

    def __init__(
        self,
        alpha=0.5,
        gamma=0.1,
        values=None,
        reward=None,
        maximum=None,
        *args,
        **kwargs,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.values = values.copy()
        self.reward = reward
        self.maximum = maximum

    def compute(self):
        """
        Compute the change in values based on the given values, reward, and parameters, and return the updated values.

        Returns
        -------
        output: numpy.ndarray:
            The computed output values.
        """

        active = self.values.copy()
        active[active > 0] = 1
        output = np.zeros(self.values.shape[0])

        for i in range(self.values.shape[0]):
            output[i] += (
                self.values[i]
                + (
                    self.alpha
                    * (self.reward + self.gamma * self.maximum - self.values[i])
                )
                * active[i]
            )

        return output

    def __repr__(self):
        return f"QLearningRule(alpha={self.alpha},\n gamma={self.gamma},\n values={self.values},\n reward={self.reward},\n maximum={self.maximum})"

    def __str__(self):
        return f"QLearningRule(alpha={self.alpha},\n gamma={self.gamma},\n values={self.values},\n reward={self.reward},\n maximum={self.maximum})"

    def __call__(self):
        return self.compute()


class HumbleTeacher:
    """
    A humble teacher learning rule (Kruschke, 1992; Love, Gureckis, and Medin, 2004) for multi-dimensional outcome learning.

    Attributes
    ----------
    alpha : float
        The learning rate.
    input : ndarray or array_like
        The input value. The stimulus representation in the form of a 1D array, where each element can take a value of 0 and 1.
    weights : ndarray
        The weights value. A 2D array of weights, where each row represents an outcome and each column represents a single stimulus.
    teacher : ndarray
        The target values or feedback, sometimes referred to as teaching signals. These are the values that the algorithm should learn to predict.
    shape : tuple
        The shape of the weight matrix.

    Parameters
    ----------
    alpha : float
        The learning rate.
    weights : array-like
        The input value. The stimulus representation in the form of a 1D array, where each element can take a value of 0 and 1.
    feedback : array-like
        The target values or feedback, sometimes referred to as teaching signals. These are the values that the algorithm should learn to predict.
    input : array-like
        The input value. The stimulus representation in the form of a 1D array, where each element can take a value of 0 and 1.
    **kwargs : dict, optional
        Additional keyword arguments.

    Notes
    -----
    The humble teacher is a learning rule that is based on the idea that if output node activations are larger than the teaching signal, they should not be counted as error, but should be rewarded. It is defined as:

    $$
    t_k = \\begin{cases}
    \\min(-1, a_k) & \\text{if } t_k = 0 \\text{ if stimulus is not followed by outcome/category-label} \\\\
    \\max(1, a_k) & \\text{if } t_k = 1 \\text{ if stimulus is followed by outcome/category-label}
    \\end{cases}
    $$

    where $t_k$ is the teaching signal. Then the change in weights is computed according to the delta rule (Rescorla & Wagner, 1972; Rumelhart, Hinton & Williams, 1986; Gluck & Bower, 1988):

    $$
    \\Delta w_{ij} = \\alpha \\cdot (t_k - a_k) \\cdot x_j
    $$

    where $\\Delta w_{ij}$ is the change in weight for the $j$-th stimulus for the $i$-th outcome, $t_k$ is the teaching signal for the $k$-th outcome, $a_k$ is the summed activation of all nodes connected to the $k$-th outcome, $x_j$ is the j-th stimulus input, and $\\alpha$ is the learning rate.

    References
    ----------
    Gluck, M. A., & Bower, G. H. (1988). From conditioning to category learning: An adaptive network model. Journal of Experimental Psychology: General, 117(3), 227–247.

    Kruschke, J. K. (1992). ALCOVE: An exemplar-based connectionist model of category learning. Psychological Review, 99, 22–44.

    Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement. In A. H. Black & W. F. Prokasy (Eds.), Classical conditioning II: Current research and theory (pp. 64-99). New York:Appleton-Century-Crofts.

    Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. nature, 323(6088), 533-536.

    Examples
    --------
    >>> import numpy as np
    >>> from cpm.models.learning import HumbleTeacher
    >>> weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    >>> teacher = np.array([0, 1])
    >>> input = np.array([1, 1, 1])
    >>> humble_teacher = HumbleTeacher(alpha=0.1, weights=weights, feedback=teacher, input=input)
    >>> humble_teacher.compute()
    array([[-0.06,  0.04,  0.14],
        [ 0.4 ,  0.5 ,  0.6 ]])
    """

    def __init__(self, alpha=None, weights=None, feedback=None, input=None, **kwargs):
        self.alpha = alpha
        self.weights = [[]]
        if weights is not None:
            self.weights = weights.copy()
        self.teacher = feedback
        self.input = np.asarray(input)
        self.shape = self.weights.shape
        self.delta = np.zeros(self.weights.shape)
        if len(self.shape) == 1:
            self.shape = (1, self.shape[0])
            self.weights = np.array([self.weights])

    def compute(self):
        """
        Compute the weights using the CPM learning rule.

        Returns
        -------
        weights: numpy.ndarray
            The updated weights matrix.
        """

        for i in range(self.shape[0]):
            activations = np.sum(self.weights[i] * self.input)
            for j in range(self.shape[1]):
                if self.teacher[i] == 0:
                    teacher = np.min([-1, activations])
                else:
                    teacher = np.max([1, activations])
                self.delta[i, j] = self.alpha * (teacher - activations) * self.input[j]
                self.weights[i, j] += self.delta[i, j]
        return self.weights


class SARSATrace:
    """
    SARSA learning rule with an eligibility trace for a two-stage Markov decision task (Sutton & Barto, 2018; Kool et al., 2016).

    Parameters
    ----------
    learning_rate : float
        The learning rate, $\\alpha$.
    eligibility_trace : float
        The eligibility trace decay, $\\lambda$. It determines how much of the second-stage prediction error is carried back to the first-stage model-free value.
    model_free_values : ndarray
        The first-stage model-free Q-values, a 2D array of shape (n_states, n_actions). The array is not modified in-place.
    second_stage_values : ndarray
        The second-stage state values, a 1D array of shape (n_second_stage_states,). The array is not modified in-place.
    starting_state : int
        The first-stage starting state (0-indexed).
    action : int
        The chosen first-stage action (0-indexed).
    reached_second_stage : int
        The second-stage state reached after the transition (0-indexed).
    reward : float
        The reward received in the second-stage state.

    Attributes
    ----------
    stage1_prediction_error : float
        The first-stage prediction error, $\\delta_1$, set after calling `compute()`.
    stage2_prediction_error : float
        The second-stage prediction error, $\\delta_2$, set after calling `compute()`.
    model_free_delta : ndarray
        The change to add to `model_free_values`, set after calling `compute()`.
    planet_value_delta : ndarray
        The change to add to `second_stage_values`, set after calling `compute()`.

    Notes
    -----
    The rule computes two coupled prediction errors. The first-stage prediction error compares the value of the reached second-stage state with the model-free value of the chosen first-stage action,

    $$
    \\delta_1 = Q_2(s_2) - Q_{MF}(s_1, a),
    $$

    and the second-stage prediction error compares the reward with the value of the reached second-stage state,

    $$
    \\delta_2 = r - Q_2(s_2).
    $$

    The values are then updated as

    $$
    \\Delta Q_{MF}(s_1, a) = \\alpha \\delta_1 + \\lambda \\alpha \\delta_2,
    $$

    $$
    \\Delta Q_2(s_2) = \\alpha \\delta_2,
    $$

    where the eligibility trace $\\lambda$ carries the second-stage prediction error back to the first-stage choice. All other values remain unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> from cpm.models.learning import SARSATrace
    >>> model_free_values = np.zeros((2, 2))
    >>> second_stage_values = np.array([4.5, 4.5])
    >>> sarsa = SARSATrace(
    ...     learning_rate=0.5, eligibility_trace=0.6,
    ...     model_free_values=model_free_values, second_stage_values=second_stage_values,
    ...     starting_state=0, action=1, reached_second_stage=0, reward=7.0,
    ... )
    >>> model_free_delta, planet_value_delta = sarsa.compute()
    >>> model_free_delta
    array([[0., 3.],
           [0., 0.]])
    >>> planet_value_delta
    array([1.25, 0.  ])

    References
    ----------
    Kool, W., Cushman, F. A., & Gershman, S. J. (2016). When does model-based control pay off? PLoS Computational Biology, 12(8), e1005090.

    Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (Second edition). The MIT Press.
    """

    def __init__(
        self,
        learning_rate=None,
        eligibility_trace=None,
        model_free_values=None,
        second_stage_values=None,
        starting_state=None,
        action=None,
        reached_second_stage=None,
        reward=None,
        **kwargs,
    ):
        self.learning_rate = learning_rate
        self.eligibility_trace = eligibility_trace
        self.model_free_values = np.asarray(model_free_values, dtype=float)
        self.second_stage_values = np.asarray(second_stage_values, dtype=float)
        self.starting_state = starting_state
        self.action = action
        self.reached_second_stage = reached_second_stage
        self.reward = reward

        self.stage1_prediction_error = None
        self.stage2_prediction_error = None
        self.model_free_delta = None
        self.planet_value_delta = None

    def compute(self):
        """
        Compute the changes in the model-free and second-stage values.

        Returns
        -------
        model_free_delta : numpy.ndarray
            The change to add to the model-free values. Non-zero only at `[starting_state, action]`.
        planet_value_delta : numpy.ndarray
            The change to add to the second-stage values. Non-zero only at `[reached_second_stage]`.
        """
        state = self.starting_state
        action = self.action
        planet = self.reached_second_stage

        self.stage1_prediction_error = (
            self.second_stage_values[planet] - self.model_free_values[state, action]
        )
        self.stage2_prediction_error = self.reward - self.second_stage_values[planet]

        self.model_free_delta = np.zeros_like(self.model_free_values)
        self.planet_value_delta = np.zeros_like(self.second_stage_values)

        self.model_free_delta[state, action] = (
            self.learning_rate * self.stage1_prediction_error
            + self.eligibility_trace
            * self.learning_rate
            * self.stage2_prediction_error
        )
        self.planet_value_delta[planet] = (
            self.learning_rate * self.stage2_prediction_error
        )

        return self.model_free_delta, self.planet_value_delta

    def __repr__(self):
        return f"SARSATrace(learning_rate={self.learning_rate},\n eligibility_trace={self.eligibility_trace},\n starting_state={self.starting_state},\n action={self.action},\n reached_second_stage={self.reached_second_stage},\n reward={self.reward})"

    def __str__(self):
        return self.__repr__()

    def __call__(self):
        return self.compute()
