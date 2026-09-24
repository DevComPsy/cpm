r"""
Function versions of the building blocks in `cpm.models`, for compiled session models.

The classes in :mod:`cpm.models.decision`, :mod:`cpm.models.learning`,
:mod:`cpm.models.activation` and :mod:`cpm.models.attention` compute one step
of a model and are convenient to write per-trial models with. This module
holds the same computations as functions on numpy arrays and floats, written
to be called *inside* a loop over trials that is compiled with numba as a
whole, such as the session models of the built-in applications (see
:class:`cpm.generators.SessionWrapper`). With numba installed, every function
here is compiled on first use (and cached on disk); without numba, they are
plain Python and give the same results.

Each function gives the same result as the class it is named after, to within
rounding error, and is tested against it. The exceptions are deliberate: the
softmax, sigmoid and logistic functions here are computed in a way that cannot
overflow, so where the classes return NaN (or replace it) because an
exponential overflowed, these return the limiting probabilities.

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
    "softmax_noise",
    "sigmoid",
    "greedy",
    "choice_kernel",
    "choose",
    "delta_rule",
    "separable_rule",
    "q_learning",
    "humble_teacher",
    "sarsa_trace_update",
    "sigmoid_activation",
    "competitive_gating",
    "prospect_utility",
    "weight_tk",
    "weight_power",
    "weight_prelec",
    "weight_gw",
    "prospect_weight",
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
    The softmax policy, :math:`e^{\beta x_i} / \sum_j e^{\beta x_j}`.

    As :class:`cpm.models.decision.Softmax` (``Softmax(temperature=beta,
    activations=values).compute()``), without overflow.

    Parameters
    ----------
    values : numpy.ndarray
        The activations, a 1D array.
    beta : float
        The inverse temperature.

    Returns
    -------
    numpy.ndarray
    """
    n = values.shape[0]
    scaled = np.empty(n)
    top = -np.inf
    for i in range(n):
        scaled[i] = values[i] * beta
        if scaled[i] > top:
            top = scaled[i]
    total = 0.0
    for i in range(n):
        scaled[i] = math.exp(scaled[i] - top)
        total += scaled[i]
    for i in range(n):
        scaled[i] /= total
    return scaled


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
def softmax_noise(values, beta, xi):
    """The softmax with irreducible noise, as `Softmax.irreducible_noise`."""
    policy = softmax(values, beta)
    n = policy.shape[0]
    for i in range(n):
        policy[i] = policy[i] * (1.0 - xi) + xi / n
    return policy


@njit
def sigmoid(activations, temperature, bias):
    """As :class:`cpm.models.decision.Sigmoid` (``Sigmoid(temperature, activations, beta=bias)``), without overflow."""
    n = activations.shape[0]
    out = np.empty(n)
    for i in range(n):
        out[i] = logistic((activations[i] - bias) * temperature)
    return out


@njit
def greedy(activations, epsilon):
    """
    As :class:`cpm.models.decision.GreedyRule`.

    Parameters
    ----------
    activations : numpy.ndarray
        A 2D array, one row per action, summed over columns.
    epsilon : float
        The exploration parameter.
    """
    n = activations.shape[0]
    summed = np.empty(n)
    for i in range(n):
        summed[i] = np.sum(activations[i])
    top = np.max(summed)
    policy = np.empty(n)
    for i in range(n):
        if summed[i] <= 0:
            policy[i] = 0.0
        elif summed[i] == top:
            policy[i] = 1.0 - (n - 1) * epsilon
        else:
            policy[i] = epsilon
    total = np.sum(policy)
    if np.all(policy == 0):
        policy[:] = 1.0 / n
    else:
        policy /= total
    return policy


@njit
def choice_kernel(activations, kernel, temperature_activations, temperature_kernel):
    """
    As :class:`cpm.models.decision.ChoiceKernel`.

    Note that `ChoiceKernel` multiplies the summed activations with the scaled
    kernel inside the exponential; this function does the same.
    """
    n = kernel.shape[0]
    out = np.empty(n)
    total = 0.0
    for i in range(n):
        out[i] = math.exp(
            np.sum(activations[i] * temperature_activations) * (kernel[i] * temperature_kernel)
        )
        total += out[i]
    for i in range(n):
        out[i] /= total
    return out


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
    As :class:`cpm.models.learning.DeltaRule`: the change in the weights and the prediction errors.

    Parameters
    ----------
    weights : numpy.ndarray
        A 2D array, one row per outcome.
    feedback : numpy.ndarray
        The teaching signal, one per outcome.
    input : numpy.ndarray
        The stimulus representation.
    alpha : float
        The learning rate.

    Returns
    -------
    change : numpy.ndarray
        The change in `weights`, with the same shape.
    error : numpy.ndarray
        The summed prediction error of each outcome.
    """
    rows, columns = weights.shape
    change = np.empty((rows, columns))
    error = np.empty(rows)
    for i in range(rows):
        activation = np.sum(weights[i] * input)
        error[i] = feedback[i] - activation
        for j in range(columns):
            change[i, j] = alpha * error[i] * input[j]
    return change, error


@njit
def separable_rule(weights, feedback, input, alpha):
    """
    As :class:`cpm.models.learning.SeparableRule`: the change in the weights and the prediction errors.

    Returns
    -------
    change : numpy.ndarray
        The change in `weights`, with the same shape.
    error : numpy.ndarray
        The prediction error of each outcome-stimulus pair, with the same shape.
    """
    rows, columns = weights.shape
    change = np.empty((rows, columns))
    error = np.empty((rows, columns))
    for i in range(rows):
        for j in range(columns):
            error[i, j] = feedback[i] - weights[i, j]
            change[i, j] = alpha * error[i, j] * input[j]
    return change, error


@njit
def q_learning(values, reward, maximum, alpha, gamma):
    """
    As :class:`cpm.models.learning.QLearningRule`: the updated values.

    Like the class, values that are not positive are not treated as active:
    the update of each value is scaled by 1 if it is positive, and by the value
    itself otherwise.
    """
    n = values.shape[0]
    out = np.empty(n)
    for i in range(n):
        active = 1.0 if values[i] > 0 else values[i]
        out[i] = values[i] + alpha * (reward + gamma * maximum - values[i]) * active
    return out


@njit
def humble_teacher(weights, feedback, input, alpha):
    """As :class:`cpm.models.learning.HumbleTeacher`: the updated weights (a new array)."""
    rows, columns = weights.shape
    out = weights.copy()
    for i in range(rows):
        activation = np.sum(out[i] * input)
        if feedback[i] == 0:
            teacher = min(-1.0, activation)
        else:
            teacher = max(1.0, activation)
        for j in range(columns):
            out[i, j] += alpha * (teacher - activation) * input[j]
    return out


@njit
def sarsa_trace_update(model_free_values, second_stage_values, starting_state, action,
                       reached_second_stage, reward, learning_rate, eligibility_trace):
    """
    As :class:`cpm.models.learning.SARSATrace`, updating the values in place.

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
    stage1 = (second_stage_values[reached_second_stage]
              - model_free_values[starting_state, action])
    stage2 = reward - second_stage_values[reached_second_stage]
    model_free_values[starting_state, action] += (
        learning_rate * stage1 + eligibility_trace * learning_rate * stage2
    )
    second_stage_values[reached_second_stage] += learning_rate * stage2
    return stage1, stage2


## ----------------------------------------------------------------------------
## activation functions (cpm.models.activation)


@njit
def sigmoid_activation(input, weights):
    """As :class:`cpm.models.activation.SigmoidActivation` for 2D weights (one row per outcome), without overflow."""
    rows, columns = weights.shape
    out = np.empty((rows, columns))
    for i in range(rows):
        for j in range(columns):
            out[i, j] = logistic(input[j] * weights[i, j])
    return out


@njit
def competitive_gating(input, values, salience, P):
    """As :class:`cpm.models.activation.CompetitiveGating`: the gated values (a new array)."""
    gain = (input * salience) ** P
    gain = gain / np.sum(gain) ** (1.0 / P)
    rows, columns = values.shape
    out = np.empty((rows, columns))
    for i in range(rows):
        for k in range(columns):
            out[i, k] = values[i, k] * gain[k]
    return out


@njit
def prospect_utility(magnitude, alpha, beta, lambda_loss):
    r"""
    The power utility of `ProspectUtility`: :math:`x^\alpha` for gains, :math:`-\lambda (-x)^\beta` for losses.
    """
    if magnitude >= 0:
        return magnitude**alpha
    return -lambda_loss * (-magnitude) ** beta


@njit
def weight_tk(probability, power):
    """The Tversky & Kahneman (1992) weighting function with curvature `power`."""
    numerator = probability**power
    return numerator / (numerator + (1.0 - probability) ** power) ** (1.0 / power)


@njit
def weight_power(probability, gamma):
    """The power weighting function, p^gamma."""
    return probability**gamma


@njit
def weight_prelec(probability, gamma, delta):
    """The Prelec (1998) weighting function."""
    if probability <= 0:
        return 0.0
    return math.exp(-delta * (-math.log(probability)) ** gamma)


@njit
def weight_gw(probability, gamma, delta):
    """The Gonzalez & Wu (1999) weighting function."""
    numerator = delta * probability**gamma
    return numerator / (numerator + (1.0 - probability) ** gamma)


## the weighting functions by number, for `prospect_weight`
WEIGHTING = {"tk": 0, "power": 1, "prelec": 2, "gw": 3}


@njit
def prospect_weight(probability, magnitude, gamma, delta, weighting):
    """
    The decision weight of an outcome, as `ProspectUtility` computes it.

    Parameters
    ----------
    probability, magnitude : float
        The probability and magnitude of the outcome.
    gamma, delta : float
        The parameters of the weighting function. With the Tversky & Kahneman
        function, `gamma` applies to gains (positive magnitudes) and `delta` to
        losses and zero magnitudes, as in `ProspectUtility`.
    weighting : int
        The weighting function: ``WEIGHTING[name]`` for name in "tk", "power",
        "prelec" and "gw".
    """
    if weighting == 0:
        return weight_tk(probability, gamma if magnitude > 0 else delta)
    if weighting == 1:
        return weight_power(probability, gamma)
    if weighting == 2:
        return weight_prelec(probability, gamma, delta)
    return weight_gw(probability, gamma, delta)


@njit
def offset(input, offset, index):
    """As :class:`cpm.models.activation.Offset`: a copy of `input` with `offset` added to element `index`."""
    out = input.copy()
    out[index] += offset
    return out


## ----------------------------------------------------------------------------
## attention (cpm.models.attention)


@njit
def rapid_attention_shift(weights, predictions, input, error, gain, pnorm, P, rho):
    """As :class:`cpm.models.attention.RapidAttentionShift`: the change in the attention gain."""
    rows, columns = weights.shape
    out = np.zeros(columns)
    for i in range(rows):
        for j in range(columns):
            out[j] += (weights[i, j] * input[j] - predictions[i] * gain[j] ** (P - 1.0)) * error[i]
    for j in range(columns):
        out[j] = rho * (pnorm**-1) * out[j]
    return out
