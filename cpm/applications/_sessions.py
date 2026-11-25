"""
The models of the built-in applications.

Each function computes all trials of one participant for one of the
applications (`RLRW`, `HybridMBMF`, `PTSM`, `PTSM1992`, `PTSM2025` and
`BayesianInformationGathering`), from plain floats and numpy arrays, using the
kernels of `cpm.models.kernels`. They
are the only implementation of these models: the per-trial `model` function of
an application runs the same loop on a single trial. With numba installed, they
are compiled as a whole; `cpm.core._jit.kernels` loads this module either
compiled or as plain Python.

Random choices are made from uniform random numbers drawn with NumPy before
the loop, one per trial, which `kernels.choose` turns into the choice that
`numpy.random.choice` would have made. Simulations are therefore reproducible
with `numpy.random.seed`, identical across backends, and identical to the
former per-trial implementations of the applications.
"""

import math

import numpy as np

from ..core import _jit

njit = _jit.decorator(globals())
kernels = _jit.kernels("cpm.models.kernels", python=globals().get(_jit.PYTHON_FLAG, False))

softmax = kernels.softmax
choose = kernels.choose
sarsa_trace_update = kernels.sarsa_trace_update
separable_rule = kernels.separable_rule
prospect_utility = kernels.prospect_utility
prospect_weight = kernels.prospect_weight
expected_utility = kernels.expected_utility


@njit
def check_activations(activations):
    """Raise ValueError for NaN or infinite activations, as `cpm.models.decision.Softmax` does."""
    if np.isfinite(activations).all():  # one call on every trial, the cheap one in plain Python
        return
    if np.isnan(activations).any():
        raise ValueError("Activations contain NaN values. Please remove or impute missing values.")
    if np.isinf(activations).any():
        raise ValueError(
            "Activations contain infinite values. Please remove or impute infinite values."
        )


@njit
def rlrw(alpha, temperature, initial_values, arms, rewards, response, generate, uniforms):
    """
    All trials of `RLRW`.

    Parameters
    ----------
    alpha, temperature : float
    initial_values : numpy.ndarray
        The initial value of each stimulus, shape (stimuli,).
    arms : numpy.ndarray
        The stimulus on each arm (1-based, as in the data), shape (trials, arms).
    rewards : numpy.ndarray
        The reward of each arm, shape (trials, arms).
    response : numpy.ndarray
        The chosen arm on each trial (0-based); ignored if `generate`.
    generate : bool
        Whether to choose from the policy instead.
    uniforms : numpy.ndarray
        One uniform random number per trial if `generate`.

    Returns
    -------
    policy, response, reward, values, change, dependent : numpy.ndarray
        `response` is the arm chosen on each trial: the sampled one if
        `generate`, otherwise the observed one.
    """
    n, k = arms.shape
    d = initial_values.shape[0]
    ## numba does not check indices, so the data are checked here, as the
    ## per-trial models did by raising IndexError
    if k < 2:
        raise ValueError("RLRW needs at least two arms.")
    values = initial_values.copy()
    policy = np.empty((n, k))
    out_response = np.empty(n, dtype=np.int64)
    reward = np.empty(n)
    history = np.empty((n, d))
    change = np.empty((n, d))
    activations = np.empty(k)
    mute = np.zeros(d)
    feedback = np.empty(1)
    for t in range(n):
        for j in range(k):
            if not -d <= arms[t, j] - 1 < d:
                raise IndexError("RLRW: the arm columns must hold stimuli from 1 to dimensions.")
            activations[j] = values[arms[t, j] - 1]
        check_activations(activations)
        policy[t] = softmax(activations, temperature)
        choice = choose(policy[t], uniforms[t]) if generate else response[t]
        if not -k <= choice < k:
            raise IndexError("RLRW: response must be an arm, from 0 to the number of arms - 1.")
        if choice < 0:
            choice += k
        out_response[t] = choice
        stimulus = arms[t, choice] - 1
        if stimulus < 0:
            stimulus += d
        teacher = rewards[t, choice]
        reward[t] = teacher
        ## the update of SeparableRule, for the chosen stimulus only
        mute[:] = 0.0
        mute[stimulus] = 1.0
        feedback[0] = teacher
        update, _ = separable_rule(values.reshape(1, d), feedback, mute, alpha)
        change[t] = update[0]
        values += update[0]
        history[t] = values
    return policy, out_response, reward, history, change, policy[:, 1].copy()


@njit
def hybrid_mbmf(inv_temperature, learning_rate, eligibility_trace, mb_weight,
                choice_stickiness, response_stickiness, q_mf_0, q2_0, m_0, r_0,
                s1, stimuli_first, action, s2, reward, position, reward_0, reward_1,
                generate, uniforms):
    """
    All trials of `HybridMBMF`.

    The data arrays are one value per trial. `action`, `s2`, `reward` and
    `position` are ignored if `generate`, and `reward_0` and `reward_1` are
    used only then. Returns the outputs of `HybridMBMF`, one row per trial.
    """
    n = s1.shape[0]
    q_mf = q_mf_0.copy()
    q2 = q2_0.copy()
    m = m_0.copy()
    r = r_0.copy()
    out_policy = np.empty((n, 2))
    out_action = np.empty(n, dtype=np.int64)
    out_s2 = np.empty(n, dtype=np.int64)
    out_reward = np.empty(n)
    out_position = np.empty(n, dtype=np.int64)
    out_pe1 = np.empty(n)
    out_pe2 = np.empty(n)
    out_q_mf = np.empty((n, 2, 2))
    out_q2 = np.empty((n, 2))
    out_m = np.empty((n, 2, 2))
    out_r = np.empty((n, 2))
    q_hybrid = np.empty(2)
    for t in range(n):
        state = s1[t]
        ## numba does not check indices, see rlrw
        if not -2 <= state < 2:
            raise IndexError("HybridMBMF: s1, stimuli_first, action, s2 and position must be 0 or 1.")
        first = stimuli_first[t]
        ## the model-based value of action 0 is that of state 1, and vice versa
        for i in range(2):
            r_action = r[1 - i] if first == 1 else r[i]
            q_hybrid[i] = (
                mb_weight * q2[1 - i]
                + (1 - mb_weight) * q_mf[state, i]
                + choice_stickiness * m[state, i]
                + response_stickiness * r_action
            )
        check_activations(q_hybrid)
        policy = softmax(q_hybrid, inv_temperature)
        if generate:
            chosen = choose(policy, uniforms[t])
            reached = 1 - chosen
            payout = reward_1[t] if reached == 1 else reward_0[t]
            side = first ^ chosen
        else:
            chosen = action[t]
            reached = s2[t]
            payout = reward[t]
            side = position[t]
        if not (-2 <= chosen < 2 and -2 <= reached < 2 and -2 <= side < 2):
            raise IndexError("HybridMBMF: s1, stimuli_first, action, s2 and position must be 0 or 1.")
        m[:, :] = 0.0
        m[state, chosen] = 1.0
        r[:] = 0.0
        r[side] = 1.0
        pe1, pe2 = sarsa_trace_update(q_mf, q2, state, chosen, reached, payout,
                                      learning_rate, eligibility_trace)
        out_policy[t] = policy
        out_action[t] = chosen
        out_s2[t] = reached
        out_reward[t] = payout
        out_position[t] = side
        out_pe1[t] = pe1
        out_pe2[t] = pe2
        out_q_mf[t] = q_mf
        out_q2[t] = q2
        out_m[t] = m
        out_r[t] = r
    return (out_policy, out_action, out_s2, out_reward, out_position, out_pe1, out_pe2,
            out_q_mf, out_q2, out_m, out_r)


@njit
def prospect_softmax(alpha, beta, lambda_loss, gamma, delta, temperature, safe, risky,
                     probability, observed, weighting, utilities, custom_utilities,
                     choose_always, generate, uniforms, dependent_chosen):
    """
    All trials of `PTSM` and `PTSM1992`: prospect-theory utilities and a softmax.

    Parameters
    ----------
    alpha, beta, lambda_loss, gamma, delta, temperature : float
    safe, risky, probability, observed : numpy.ndarray
        One value per trial.
    weighting : int
        The weighting function, see `kernels.WEIGHTING`.
    utilities : numpy.ndarray
        The utilities of the safe and risky magnitude on each trial, shape
        (trials, 2), if `custom_utilities` (computed by a user-supplied
        utility curve); otherwise ignored.
    choose_always : bool
        Whether a choice is sampled on every trial (as `PTSM1992` does), or only
        if `generate` (as `PTSM` does).
    dependent_chosen : bool
        Whether the dependent variable is the probability of the observed choice
        (`PTSM`) rather than of the risky option (`PTSM1992`).

    Returns
    -------
    policy, dependent, chosen, is_optimal, objective_best, ev_safe, ev_risk, u_safe, u_risk
    """
    n = safe.shape[0]
    out_policy = np.empty((n, 2))
    out_dependent = np.empty(n)
    out_chosen = np.empty(n, dtype=np.int64)
    out_optimal = np.empty(n, dtype=np.int64)
    out_best = np.empty(n, dtype=np.int64)
    out_ev_risk = np.empty(n)
    out_u_safe = np.empty(n)
    out_u_risk = np.empty(n)
    ## the expected utilities of the two options (safe, risky) on all trials at
    ## once, as ProspectUtility computes them for options with one outcome each;
    ## they do not depend on earlier trials. The safe option pays with probability 1.
    magnitudes = np.empty((n, 2))
    magnitudes[:, 0] = safe
    magnitudes[:, 1] = risky
    probabilities = np.ones((n, 2))
    probabilities[:, 1] = probability
    weights = prospect_weight(probabilities, magnitudes, gamma, delta, weighting)
    if custom_utilities:
        outcome_utilities = utilities
    else:
        outcome_utilities = prospect_utility(magnitudes, alpha, beta, lambda_loss)
    expected_all = expected_utility(
        weights.reshape(2 * n, 1), outcome_utilities.reshape(2 * n, 1)
    ).reshape(n, 2)
    for t in range(n):
        ev_risk = risky[t] * probability[t]
        best = 1 if ev_risk >= safe[t] else 0
        expected = expected_all[t]
        check_activations(expected)
        policy = softmax(expected, temperature)
        ## numba does not check indices, see rlrw
        if (dependent_chosen or not (choose_always or generate)) and not -2 <= observed[t] < 2:
            raise IndexError("observed must be 0 (safe) or 1 (risky).")
        if choose_always or generate:
            chosen = choose(policy, uniforms[t])
        else:
            chosen = observed[t]
        out_policy[t] = policy
        out_dependent[t] = policy[observed[t]] if dependent_chosen else policy[1]
        out_chosen[t] = chosen
        out_optimal[t] = 1 if chosen == best else 0
        out_best[t] = best
        out_ev_risk[t] = ev_risk
        out_u_safe[t] = expected[0]
        out_u_risk[t] = expected[1]
    return (out_policy, out_dependent, out_chosen, out_optimal, out_best, out_ev_risk,
            out_u_safe, out_u_risk)


@njit
def power_utility(x, alpha):
    """The piecewise power utility of `PTSM2025`, which its parameters also hold as `utility_curvature`."""
    return x ** alpha if x >= 0 else -np.abs(x) ** alpha


@njit
def ptsm2025(eta, phi_gain, phi_loss, temperature, alpha, safe, risky, probability,
             ambiguity, uniforms):
    """
    All trials of `PTSM2025`.

    Returns
    -------
    policy, model_choice, u_safe, u_risk : numpy.ndarray
    """
    n = safe.shape[0]
    out_policy = np.empty(n)
    out_choice = np.empty(n, dtype=np.int64)
    out_u_safe = np.empty(n)
    out_u_risk = np.empty(n)
    pair = np.empty(2)
    terms = np.empty(2)
    for t in range(n):
        subjective = probability[t] - eta * ambiguity[t]
        subjective = min(max(subjective, 0.0), 1.0)
        u_safe = power_utility(safe[t], alpha)
        u_risk = subjective * power_utility(risky[t], alpha)
        phi = phi_gain if risky[t] >= 0 else phi_loss
        ## the policy of PTSM2025: a softmax of the scaled utilities, with the
        ## gambling bias added to the risky one
        terms[0] = temperature * u_safe
        terms[1] = temperature * u_risk + phi
        p = softmax(terms, 1.0)[1]
        pair[0] = 1 - p
        pair[1] = p
        out_policy[t] = p
        out_choice[t] = choose(pair, uniforms[t])
        out_u_safe[t] = u_safe
        out_u_risk[t] = u_risk
    return out_policy, out_choice, out_u_safe, out_u_risk


@njit
def information_gathering_policy(belief, transitions, cost_scale, cost_slope, impatience,
                                 temperature, lapse_rate, r_correct, r_incorrect, r_none,
                                 policy, q_open, cost):
    """
    The policy of `BayesianInformationGathering` in one condition, by backward induction.

    Fills `policy[N, n_y]` (deciding yellow, deciding blue, opening another card),
    `q_open[N, n_y]` (the value of opening another card) and `cost[N]` (its
    cost) for every state of the game, from the last card back to the first, as
    the code of the authors computes them.

    Parameters
    ----------
    belief : numpy.ndarray
        The probability that yellow is the right answer in each state, shape
        (cards + 1, cards + 1), indexed by the cards seen and the yellow ones among them.
    transitions : numpy.ndarray
        The probability that the next card is blue (0) or yellow (1) in each
        state, shape (cards + 1, cards + 1, 2).
    cost_scale, cost_slope, impatience, temperature, lapse_rate : float
    r_correct, r_incorrect, r_none : float
        The value of a right and of a wrong answer, and of not answering after the last card.
    policy, q_open, cost : numpy.ndarray
        Filled in place; shapes (cards + 1, cards + 1, 3), (cards + 1, cards + 1)
        and (cards + 1,). Row 0, before the first card, is left as it is.
    """
    cards = belief.shape[0] - 1
    value = np.empty(cards + 1)
    following = np.empty(cards + 1)
    inverse = 1.0 / temperature
    noise = lapse_rate / 3.0
    for n in range(cards, 0, -1):
        ## the sigmoid cost of opening another card after n cards
        c_s = cost_scale / (1.0 + math.exp(-cost_slope * (n - impatience)))
        cost[n] = c_s
        for k in range(n + 1):
            p = belief[n, k]
            q_yellow = r_correct * p + r_incorrect * (1.0 - p)
            q_blue = r_correct * (1.0 - p) + r_incorrect * p
            if n == cards:
                ## no card is left to open
                q_more = -c_s + r_none
            else:
                q_more = -c_s + (transitions[n, k, 0] * following[k]
                                 + transitions[n, k, 1] * following[k + 1])
            ## the softmax with a lapse rate, written out for three actions so
            ## that it allocates nothing; the largest value is subtracted first,
            ## as in the code of the authors, so that no value overflows, and
            ## its exponential is 1
            if q_yellow >= q_blue and q_yellow >= q_more:
                e_yellow = 1.0
                e_blue = math.exp((q_blue - q_yellow) * inverse)
                e_more = math.exp((q_more - q_yellow) * inverse)
            elif q_blue >= q_more:
                e_yellow = math.exp((q_yellow - q_blue) * inverse)
                e_blue = 1.0
                e_more = math.exp((q_more - q_blue) * inverse)
            else:
                e_yellow = math.exp((q_yellow - q_more) * inverse)
                e_blue = math.exp((q_blue - q_more) * inverse)
                e_more = 1.0
            scale = (1.0 - lapse_rate) / (e_yellow + e_blue + e_more)
            pi_yellow = e_yellow * scale + noise
            pi_blue = e_blue * scale + noise
            pi_more = e_more * scale + noise
            policy[n, k, 0] = pi_yellow
            policy[n, k, 1] = pi_blue
            policy[n, k, 2] = pi_more
            q_open[n, k] = q_more
            value[k] = q_yellow * pi_yellow + q_blue * pi_blue + q_more * pi_more
        value, following = following, value


@njit
def information_gathering(cost_scale, cost_slope, impatience_fixed, impatience_decreasing,
                          temperature, lapse_rate, belief, transitions, r_correct,
                          r_incorrect, r_none, n_yellow, n_seen, condition, action,
                          generate, uniforms):
    """
    All trials of `BayesianInformationGathering`, one per state of a game.

    Parameters
    ----------
    cost_scale, cost_slope, impatience_fixed, impatience_decreasing, temperature, lapse_rate : float
    belief, transitions, r_correct, r_incorrect, r_none
        See `information_gathering_policy`.
    n_yellow, n_seen, condition, action : numpy.ndarray
        One value per trial; `action` is ignored if `generate`.
    generate : bool
        Whether to play the games instead; the trials must then hold every card
        of each game, see `BayesianInformationGathering`.
    uniforms : numpy.ndarray
        One uniform random number per trial if `generate`.

    Returns
    -------
    policy, action, belief, q_values, cost, dependent : numpy.ndarray
        `action` is the observed action, or the sampled one if `generate`, and -1
        on the cards of a game after the agent decided; `dependent` is the
        probability of `action`, and NaN where it is -1.
    """
    n = n_seen.shape[0]
    cards = belief.shape[0] - 1
    ## numba does not check indices, so the data are checked here, see rlrw
    used = np.zeros(2, dtype=np.bool_)
    for t in range(n):
        if not 1 <= n_seen[t] <= cards:
            raise IndexError(
                "BayesianInformationGathering: n_seen must be from 1 to the number of cards (25)."
            )
        if not 0 <= n_yellow[t] <= n_seen[t]:
            raise IndexError("BayesianInformationGathering: n_yellow must be from 0 to n_seen.")
        if not 0 <= condition[t] <= 1:
            raise IndexError(
                "BayesianInformationGathering: condition must be 0 (fixed) or 1 (decreasing)."
            )
        if not generate and not 0 <= action[t] <= 2:
            raise IndexError(
                "BayesianInformationGathering: action must be 0 (yellow), 1 (blue) or 2 (open another card)."
            )
        used[condition[t]] = True
    if generate:
        for t in range(n):
            if n_seen[t] != 1 and (
                t == 0
                or n_seen[t] != n_seen[t - 1] + 1
                or not 0 <= n_yellow[t] - n_yellow[t - 1] <= 1
                or condition[t] != condition[t - 1]
            ):
                raise ValueError(
                    "BayesianInformationGathering: to generate, each game must start at n_seen 1 "
                    "and reveal one card per trial, in one condition."
                )
            if (t == n - 1 or n_seen[t + 1] == 1) and n_seen[t] != cards:
                raise ValueError(
                    "BayesianInformationGathering: to generate, each game must hold all 25 cards."
                )
    ## the policy of a condition is computed only if the data have trials in it
    policy_table = np.zeros((2, cards + 1, cards + 1, 3))
    q_open_table = np.zeros((2, cards + 1, cards + 1))
    cost_table = np.zeros((2, cards + 1))
    for c in range(2):
        if used[c]:
            information_gathering_policy(
                belief, transitions, cost_scale, cost_slope,
                impatience_fixed if c == 0 else impatience_decreasing, temperature,
                lapse_rate, r_correct, r_incorrect, r_none, policy_table[c], q_open_table[c],
                cost_table[c],
            )
    out_policy = np.empty((n, 3))
    out_action = np.empty(n, dtype=np.int64)
    out_belief = np.empty(n)
    out_q = np.empty((n, 3))
    out_cost = np.empty(n)
    out_dependent = np.empty(n)
    playing = False
    for t in range(n):
        c, seen, k = condition[t], n_seen[t], n_yellow[t]
        policy = policy_table[c, seen, k]
        p = belief[seen, k]
        out_policy[t] = policy
        out_belief[t] = p
        out_q[t, 0] = r_correct * p + r_incorrect * (1.0 - p)
        out_q[t, 1] = r_correct * (1.0 - p) + r_incorrect * p
        out_q[t, 2] = q_open_table[c, seen, k]
        out_cost[t] = cost_table[c, seen]
        if generate:
            if seen == 1:
                playing = True
            if not playing:
                out_action[t] = -1
                out_dependent[t] = np.nan
                continue
            ## after the last card, the agent has to decide
            if seen == cards:
                chosen = choose(policy[:2], uniforms[t])
            else:
                chosen = choose(policy, uniforms[t])
            playing = chosen == 2
        else:
            chosen = action[t]
        out_action[t] = chosen
        out_dependent[t] = policy[chosen]
    return out_policy, out_action, out_belief, out_q, out_cost, out_dependent
