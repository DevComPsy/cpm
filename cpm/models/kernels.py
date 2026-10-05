r"""
The computations of the building blocks in `cpm.models`, as functions.

The classes in :mod:`cpm.models.decision`, :mod:`cpm.models.learning`,
:mod:`cpm.models.activation` and :mod:`cpm.models.attention` compute one step
of a model and are convenient to write per-trial models with. Their formulas
live here, once, as functions on numpy arrays and numbers, and the classes
compute with them: each class converts and checks its input, calls the function
it is named after, and keeps the result in its attributes.

The functions are written with numpy operations that numba can also compile, so
that loops over trials that are compiled as a whole, such as the models of the
built-in applications (see :class:`cpm.generators.SessionWrapper`), can call
them too. With numba installed, the functions of this module are compiled on
first use (and cached on disk); the classes use a plain-Python copy of them
(see `cpm.core._jit.kernels`), which never needs numba. Both give the same
results.

`logistic`, `p_second`, `log_softmax` and `choose` have no class counterpart:
they are for compiled loops. The softmax switches to an equivalent form that
cannot overflow where the largest scaled activation is beyond ±700; `logistic`
and `p_second` cannot overflow.

Functions that update values in place say so; all others return new arrays.

Examples
--------
A Rescorla-Wagner learner with a softmax policy, compiled as a whole:

>>> import numpy as np
>>> from numba import njit
>>> from cpm.models import kernels
>>> @njit
... def session(alpha, beta, rewards, choices):
...     values = np.zeros(2)
...     p_right = np.empty(len(choices))
...     for t in range(len(choices)):
...         p_right[t] = kernels.p_second(beta, values[0], values[1])
...         values[choices[t]] += alpha * (rewards[t] - values[choices[t]])
...     return p_right
>>> session(0.3, 2.0, np.array([1.0, 0.0, 1.0]), np.array([1, 1, 0]))
array([0.5       , 0.64565631, 0.60348325])
"""

import math

import numpy as np

from ..core import _jit

njit = _jit.decorator(globals())

__all__ = [
    "logistic",
    "p_second",
    "softmax",
    "log_softmax",
    "irreducible_noise",
    "softmax_noise",
    "sigmoid",
    "greedy",
    "choice_kernel",
    "choose",
    "delta_rule",
    "separable_rule",
    "q_learning",
    "humble_teacher_change",
    "humble_teacher",
    "sarsa_trace",
    "sarsa_trace_update",
    "sigmoid_activation",
    "gating_gain",
    "gate",
    "competitive_gating",
    "prospect_utility",
    "weight_tk",
    "weight_power",
    "weight_prelec",
    "weight_gw",
    "prospect_weight",
    "expected_utility",
    "add_offset",
    "offset",
    "rapid_attention_shift",
]


## ----------------------------------------------------------------------------
## decision rules (cpm.models.decision)


@njit
def logistic(x):
    """The logistic function, 1 / (1 + exp(-x)), without overflow."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@njit
def p_second(beta, value_0, value_1):
    """
    The probability of the second of two options under a softmax.

    Equals ``softmax(np.array([value_0, value_1]), beta)[1]``, computed as the
    logistic of the scaled value difference.
    """
    return logistic(beta * (value_1 - value_0))


@njit
def softmax(values, beta):
    r"""
    The softmax policy, :math:`e^{\beta x_i} / \sum_j e^{\beta x_j}`, of :class:`cpm.models.decision.Softmax`.

    The exponentials of the scaled activations are divided by their sum. Where
    the largest scaled activation is beyond ±700, and the exponentials would
    overflow (or all underflow), the largest one is subtracted first, which gives
    the same function without overflow.

    Parameters
    ----------
    values : numpy.ndarray
        The activations, a 1D array.
    beta : float
        The inverse temperature.

    Returns
    -------
    numpy.ndarray
        The policy. If the scaled activations contain NaN, it is NaN throughout,
        which `Softmax` then replaces.
    """
    scaled = values * beta
    top = np.max(scaled) if scaled.size else 0.0
    if -700.0 < top < 700.0:
        output = np.exp(scaled)
    else:
        output = np.exp(scaled - top)
    output /= output.sum()
    return output


@njit
def log_softmax(values, beta):
    """The logarithm of `softmax`, computed without forming the probabilities."""
    n = values.shape[0]
    out = np.empty(n)
    top = -np.inf
    for i in range(n):
        out[i] = values[i] * beta
        if out[i] > top:
            top = out[i]
    total = 0.0
    for i in range(n):
        total += math.exp(out[i] - top)
    norm = top + math.log(total)
    for i in range(n):
        out[i] -= norm
    return out


@njit
def irreducible_noise(policy, xi):
    """A policy with irreducible noise `xi`, as `Softmax.irreducible_noise` computes it."""
    return policy * (1 - xi) + (xi / policy.shape[0])


@njit
def softmax_noise(values, beta, xi):
    """The softmax with irreducible noise, as `Softmax.irreducible_noise`."""
    return irreducible_noise(softmax(values, beta), xi)


@njit
def sigmoid(activations, temperature, bias):
    """The policy of :class:`cpm.models.decision.Sigmoid` (``Sigmoid(temperature, activations, beta=bias)``)."""
    return 1 / (1 + np.exp((activations - bias) * -temperature))


@njit
def greedy(activations, epsilon):
    """
    The policy of :class:`cpm.models.decision.GreedyRule`.

    Parameters
    ----------
    activations : numpy.ndarray
        A 2D array, one row per action, summed over columns.
    epsilon : float
        The exploration parameter.
    """
    output = np.sum(activations, axis=1)
    policies = np.zeros(output.shape)
    maximum = np.max(output)
    policies[output != maximum] = epsilon * 1
    policies[output == maximum] = 1 - (output.shape[0] - 1) * epsilon
    policies[output <= 0] = 0
    if np.all(policies == 0):
        policies.fill(1 / policies.shape[0])
    else:
        policies = policies / policies.sum()  # normalise
    return policies


@njit
def choice_kernel(activations, kernel, temperature_activations, temperature_kernel):
    """
    The policy of :class:`cpm.models.decision.ChoiceKernel`.

    Note that `ChoiceKernel` multiplies the summed activations with the scaled
    kernel inside the exponential.
    """
    values = activations * temperature_activations
    kernels = kernel * temperature_kernel
    nominator = np.exp(np.sum(values, axis=1) * kernels)
    denominator = np.sum(np.exp(np.sum(values, axis=1) * kernels))
    return nominator / denominator


@njit
def choose(policy, uniform):
    """
    Sample an option from a policy, given a uniform random number in [0, 1).

    This is the inverse-CDF step of ``numpy.random.choice(len(policy),
    p=policy)``, which is how the `choice()` methods of the decision classes
    sample. With ``uniform = numpy.random.random_sample()``, it returns the
    option `numpy.random.choice` would have returned, so simulations can draw
    all uniforms with NumPy before a compiled loop and still be reproducible
    with `numpy.random.seed`, and identical to the per-trial models.
    """
    n = policy.shape[0]
    cdf = np.empty(n)
    total = 0.0
    for i in range(n):
        total += policy[i]
        cdf[i] = total
    last = cdf[n - 1]
    index = 0
    for i in range(n):
        if cdf[i] / last <= uniform:
            index += 1
    if index > n - 1:
        index = n - 1
    return index


## ----------------------------------------------------------------------------
## learning rules (cpm.models.learning)


@njit
def delta_rule(weights, feedback, input, alpha):
    """
    The change in the weights and the prediction errors of :class:`cpm.models.learning.DeltaRule`.

    Parameters
    ----------
    weights : numpy.ndarray
        A 2D array, one row per outcome.
    feedback : numpy.ndarray
        The teaching signal, one per outcome.
    input : numpy.ndarray
        The stimulus representation, one value per column of `weights`.
    alpha : float
        The learning rate.

    Returns
    -------
    change : numpy.ndarray
        The change in `weights`, with the same shape.
    error : numpy.ndarray
        The summed prediction error of each outcome.
    """
    activations = np.sum(weights * input, axis=1)
    error = feedback - activations
    return alpha * error[:, np.newaxis] * input, error


@njit
def separable_rule(weights, feedback, input, alpha):
    """
    The change in the weights and the prediction errors of :class:`cpm.models.learning.SeparableRule`.

    Returns
    -------
    change : numpy.ndarray
        The change in `weights`, with the same shape.
    error : numpy.ndarray
        The prediction error of each outcome-stimulus pair, with the same shape.
    """
    error = feedback[:, np.newaxis] - weights
    return alpha * error * input, error


@njit
def q_learning(values, reward, maximum, alpha, gamma):
    """
    The updated values of :class:`cpm.models.learning.QLearningRule`.

    Like the class, values that are not positive are not treated as active:
    the update of each value is scaled by 1 if it is positive, and by the value
    itself otherwise.
    """
    active = values.copy()
    active[active > 0] = 1
    output = np.zeros(values.shape[0])
    output += values + (alpha * (reward + gamma * maximum - values)) * active
    return output


@njit
def humble_teacher_change(weights, feedback, input, alpha):
    """The change in the weights of :class:`cpm.models.learning.HumbleTeacher`."""
    activations = np.sum(weights * input, axis=1)
    teacher = np.where(feedback == 0, np.minimum(-1, activations), np.maximum(1, activations))
    return alpha * (teacher - activations)[:, np.newaxis] * input


@njit
def humble_teacher(weights, feedback, input, alpha):
    """The updated weights (a new array) of :class:`cpm.models.learning.HumbleTeacher`."""
    return weights + humble_teacher_change(weights, feedback, input, alpha)


@njit
def sarsa_trace(model_free_values, second_stage_values, starting_state, action,
                reached_second_stage, reward, learning_rate, eligibility_trace):
    """
    The prediction errors and value changes of :class:`cpm.models.learning.SARSATrace`.

    Returns
    -------
    stage1_prediction_error, stage2_prediction_error : float
    model_free_change : float
        The change of ``model_free_values[starting_state, action]``.
    second_stage_change : float
        The change of ``second_stage_values[reached_second_stage]``.
    """
    stage1 = (second_stage_values[reached_second_stage]
              - model_free_values[starting_state, action])
    stage2 = reward - second_stage_values[reached_second_stage]
    return (
        stage1,
        stage2,
        learning_rate * stage1 + eligibility_trace * learning_rate * stage2,
        learning_rate * stage2,
    )


@njit
def sarsa_trace_update(model_free_values, second_stage_values, starting_state, action,
                       reached_second_stage, reward, learning_rate, eligibility_trace):
    """
    `sarsa_trace`, applied to the values in place.

    Parameters
    ----------
    model_free_values : numpy.ndarray
        The first-stage model-free values, shape (states, actions). Updated in place.
    second_stage_values : numpy.ndarray
        The second-stage values, a 1D array. Updated in place.
    starting_state, action, reached_second_stage : int
    reward, learning_rate, eligibility_trace : float

    Returns
    -------
    stage1_prediction_error, stage2_prediction_error : float
    """
    stage1, stage2, model_free_change, second_stage_change = sarsa_trace(
        model_free_values, second_stage_values, starting_state, action,
        reached_second_stage, reward, learning_rate, eligibility_trace,
    )
    model_free_values[starting_state, action] += model_free_change
    second_stage_values[reached_second_stage] += second_stage_change
    return stage1, stage2


## ----------------------------------------------------------------------------
## activation functions (cpm.models.activation)


@njit
def sigmoid_activation(input, weights):
    """The activations of :class:`cpm.models.activation.SigmoidActivation`."""
    return 1 / (1 + np.exp(-input * weights))


@njit
def gating_gain(input, salience, P):
    """The normalised attentional gain of :class:`cpm.models.activation.CompetitiveGating`."""
    gain = input * salience
    gain = gain**P
    return gain / np.sum(gain) ** (1 / P)


@njit
def gate(values, gain):
    """The values of each stimulus scaled by its attentional gain (a new array)."""
    return values * gain[: values.shape[1]]


@njit
def competitive_gating(input, values, salience, P):
    """The gated values (a new array) of :class:`cpm.models.activation.CompetitiveGating`."""
    return gate(values, gating_gain(input, salience, P))


@njit
def prospect_utility(magnitude, alpha, beta, lambda_loss):
    r"""
    The power utility of `ProspectUtility`: :math:`x^\alpha` for gains, :math:`-\lambda (-x)^\beta` for losses.

    `magnitude` is a number or an array of any shape.
    """
    gains = magnitude >= 0
    return np.where(gains, 1.0, -lambda_loss) * np.power(
        np.abs(magnitude), np.where(gains, alpha, beta)
    )


@njit
def weight_tk(probability, power):
    """The Tversky & Kahneman (1992) weighting function with curvature `power`."""
    numerator = np.power(probability, power)
    denominator = np.power(numerator + np.power(1 - probability, power), 1 / power)
    return numerator / denominator


@njit
def weight_power(probability, gamma):
    """The power weighting function, p^gamma."""
    return np.power(probability, gamma)


@njit
def weight_prelec(probability, gamma, delta):
    """The Prelec (1998) weighting function."""
    return np.exp(-delta * np.power(-np.log(probability), gamma))


@njit
def weight_gw(probability, gamma, delta):
    """The Gonzalez & Wu (1999) weighting function."""
    numerator = delta * np.power(probability, gamma)
    denominator = numerator + np.power(1 - probability, gamma)
    return numerator / denominator


## the weighting functions by number, for `prospect_weight`
WEIGHTING = {"tk": 0, "power": 1, "prelec": 2, "gw": 3}


@njit
def prospect_weight(probability, magnitude, gamma, delta, weighting):
    """
    The decision weights of outcomes, as `ProspectUtility` computes them.

    Parameters
    ----------
    probability, magnitude : float or numpy.ndarray
        The probabilities and magnitudes of the outcomes, numbers or arrays of the
        same shape.
    gamma, delta : float
        The parameters of the weighting function. With the Tversky & Kahneman
        function, `gamma` applies to gains (positive magnitudes) and `delta` to
        losses and zero magnitudes, as in `ProspectUtility`.
    weighting : int
        The weighting function: ``WEIGHTING[name]`` for name in "tk", "power",
        "prelec" and "gw".
    """
    if weighting == 0:
        return weight_tk(probability, np.where(magnitude > 0, gamma, delta))
    if weighting == 1:
        return weight_power(probability, gamma)
    if weighting == 2:
        return weight_prelec(probability, gamma, delta)
    return weight_gw(probability, gamma, delta)


@njit
def expected_utility(weights, utilities):
    """
    The expected utility of each option: the sum of its weighted utilities.

    `weights` and `utilities` have one row per option (or one number per option,
    for options with a single outcome).
    """
    weighted = weights * utilities
    return weighted.reshape(weighted.shape[0], -1).sum(axis=1)


@njit
def add_offset(array, offset, index):
    """Add `offset` to element `index` of `array`, in place, as :class:`cpm.models.activation.Offset` does."""
    array[index] += offset
    return array


@njit
def offset(input, offset, index):
    """The output of :class:`cpm.models.activation.Offset`: a copy of `input` with `offset` added to element `index`."""
    return add_offset(input.copy(), offset, index)


## ----------------------------------------------------------------------------
## attention (cpm.models.attention)


@njit
def rapid_attention_shift(weights, predictions, input, error, gain, pnorm, P, rho):
    """The change in the attention gain of :class:`cpm.models.attention.RapidAttentionShift`."""
    activations = weights * input
    a_power = gain ** (P - 1)
    attention = np.outer(predictions, a_power)
    out = (activations - attention) * error[:, np.newaxis]
    out_sum = out.sum(axis=0)
    return rho * (pnorm**-1) * out_sum
