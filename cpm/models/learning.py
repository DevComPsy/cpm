import numpy as np

__all__ = ["DeltaRule", "SeparableRule", "QLearningRule", "KernelUpdate"]


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
        self.error = np.zeros(self.weights.shape[0])
        self.teacher = feedback
        self.input = np.asarray(input)
        self.shape = self.weights.shape
        if len(self.shape) == 1:
            self.shape = (1, self.shape[0])
            self.weights = np.array([self.weights])
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
        self.error = np.zeros(self.weights.shape[0])
        self.teacher = feedback
        self.input = np.asarray(input)
        self.shape = self.weights.shape
        if len(self.shape) == 1:
            self.shape = (1, self.shape[0])
            self.weights = np.array([self.weights])
        self.__run__ = False

    def compute(self):
        """
        Computes the prediction error using the learning rule.

        Returns:
        --------
        ndarray
            The prediction error for each stimuli-outcome mapping.
            It has the same shape as the weights input argument.
        """
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.weights[i, j] = (
                    self.alpha * (self.teacher[i] - self.weights[i, j]) * self.input[j]
                )
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

        Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(s', a')) - Q(s, a)),

    where `Q(s, a)` is the value of action `a` in state `s`, `r` is the reward received on the current state, `gamma` is the discount factor, and `max(Q(s', a'))` is the maximum estimated reward for the next state.

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


class KernelUpdate:
    """
    A class representing a learning rule for updating the choice kernel as specified by Equation 5 in Wilson and Collins (2019).

    Parameters
    ----------
    response : ndarray
        The response vector. It must be a binary numpy.ndarray, so that each element corresponds to a response option. If there are 4 response options, and the second was selected, it would be represented as `[0, 1, 0, 0]`.
    alpha : float
        The kernel learning rate.
    kernel : ndarray
        The kernel used for learning. It is a 1D array of kernel values, where each element corresponds to a response option. Each element must correspond to the same response option in the `response` vector.

    Notes
    -----
    The kernel update component is used to represent how likely a given response is to be chosen based on the frequency it was chosen in the past.
    This can then be integrated into a choice kernel decision policy.

    See Also
    --------
    [cpm.models.decision.ChoiceKernel][cpm.models.decision.ChoiceKernel] : A class representing a choice kernel decision policy.

    References
    ----------
    Wilson, Robert C., and Anne GE Collins. Ten simple rules for the computational modeling of behavioral data. Elife 8 (2019): e49547.

    """

    def __init__(self, response, alpha, kernel, input, **kwargs):
        if len(response) != len(kernel):
            raise ValueError(
                "The response and kernel must have the same number of elements."
            )
        self.response = response
        self.alpha = alpha
        self.kernel = kernel.copy()
        self.input = input

    def compute(self):
        """
        Compute the change in the kernel based on the given response, rate, and kernel, and return the updated kernel.

        Returns
        -------
        output: numpy.ndarray:
            The computed change of the kernel.
        """
        out = self.alpha * (self.response - self.kernel) * self.input
        return out

    def config(self):
        """
        Get the configuration of the kernel update component.

        Returns
        -------
        config: dict
            A dictionary containing the configuration parameters of the kernel update component.

            - response (float): The response of the system.
            - rate (float): The learning rate.
            - kernel (list): The kernel used for learning.
            - input (str): The name of the input.
            - name (str): The name of the kernel update component class.
            - type (str): The type of the kernel update component.
        """
        config = {
            "response": self.response,
            "rate": self.rate,
            "kernel": self.kernel,
            "input": self.input,
            "name": self.__class__.__name__,
            "type": "learning",
        }
        return config

    def __repr__(self):
        return f"KernelUpdate(response={self.response},\n rate={self.rate},\n kernel={self.kernel},\n input={self.input})"

    def __str__(self):
        return f"KernelUpdate(response={self.response},\n rate={self.rate},\n kernel={self.kernel},\n input={self.input})"

    def __call__(self):
        return self.compute()


class HumbleTeacher:
    """
    A humbe teacher learning rule (Kruschke, 1992; Love, Gureckis, and Medin, 2004) for multi-dimensional outcome learning.

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
