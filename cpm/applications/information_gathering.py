import functools
import numbers
import warnings

import numpy
from scipy.stats import beta, betabinom

from cpm.generators import Parameters, Value
from cpm.applications._backend import Application, SessionModel, indices, prepared, require, uniforms

## the information sampling task: 25 cards, of which 13 make a majority
CARDS = 25
MAJORITY = 13
## the prior of the generative probability of yellow cards, Beta(1, 1)
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
## the value of a right and of a wrong answer, and of not answering after the last card
R_CORRECT = 100.0
R_INCORRECT = -100.0
R_NONE = -10000.0

BELIEFS = ("majority", "generative")

PARAMETERS = {
    # name: ([initial, lower, upper], prior)
    "cost_scale": ([10.0, 0.0, 30.0], {"mean": 10.0, "sd": 5.0}),
    "cost_slope": ([10.0, 0.0, 20.0], {"mean": 10.0, "sd": 5.0}),
    "impatience_fixed": ([12.5, 0.0, 25.0], {"mean": 12.5, "sd": 6.25}),
    "impatience_decreasing": ([12.5, 0.0, 25.0], {"mean": 12.5, "sd": 6.25}),
    "temperature": ([5.0, 1.0, 50.0], {"mean": 5.0, "sd": 10.0}),
    "lapse_rate": ([0.05, 0.0, 0.5], {"mean": 0.05, "sd": 0.1}),
}


def _parameters(parameters_settings):
    """The parameters of `BayesianInformationGathering`."""
    if parameters_settings is None:
        parameters_settings = {}
        warnings.warn("No parameters specified, using default settings.", stacklevel=3)
    unknown = sorted(set(parameters_settings) - set(PARAMETERS))
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {unknown}; the parameters of BayesianInformationGathering "
            f"are {list(PARAMETERS)}."
        )
    values = {}
    for name, (default, prior) in PARAMETERS.items():
        settings = parameters_settings.get(name, default)
        if isinstance(settings, numbers.Real):
            ## a single number fixes the parameter, which is then not fitted
            values[name] = float(settings)
            continue
        values[name] = Value(
            value=settings[0],
            lower=settings[1],
            upper=settings[2],
            prior="truncated_normal",
            args=prior,
        )
    # freely varying parameters are indicated by specifying priors
    return Parameters(**values)


@functools.lru_cache(maxsize=None)
def _tables(belief):
    """
    The belief and the transition probabilities in each state, which do not depend on the parameters.

    Returns
    -------
    belief : numpy.ndarray
        The probability that yellow is the right answer, P(MY | n_y, N) or
        P(q > 1/2 | n_y, N), shape (cards + 1, cards + 1), indexed by N and n_y.
    transitions : numpy.ndarray
        The probability that the next card is blue (0) or yellow (1), shape
        (cards + 1, cards + 1, 2).
    """
    table = numpy.zeros((CARDS + 1, CARDS + 1))
    transitions = numpy.zeros((CARDS + 1, CARDS + 1, 2))
    for seen in range(1, CARDS + 1):
        yellow = numpy.arange(seen + 1)
        blue = seen - yellow
        ## the posterior of the generative probability q (Equation 1.7)
        a, b = PRIOR_ALPHA + yellow, PRIOR_BETA + blue
        if belief == "majority":
            ## a majority already seen decides the answer (Equation 1.2) ...
            p = numpy.where(yellow >= MAJORITY, 1.0, 0.0)
            ## ... otherwise, the remaining cards are a beta-binomial draw (Equation 1.3)
            open_ = (yellow < MAJORITY) & (blue < MAJORITY)
            if open_.any():
                needed = MAJORITY - yellow[open_]
                p[open_] = betabinom.sf(needed - 1, CARDS - seen, a[open_], b[open_])
        else:
            p = beta.sf(0.5, a, b)
        table[seen, : seen + 1] = p
        ## the posterior mean of q (Equation 1.11)
        transitions[seen, : seen + 1, 0] = b / (a + b)
        transitions[seen, : seen + 1, 1] = a / (a + b)
    table.flags.writeable = False
    transitions.flags.writeable = False
    return table, transitions


class _InformationModel(SessionModel):
    """The model function of `BayesianInformationGathering`, for all trials at once."""

    def __init__(self, belief, generate):
        super().__init__(generate)
        self.belief = belief

    def __call__(self, parameters, data):
        belief, transitions = _tables(self.belief)
        trials = data["n_seen"].shape[0]
        policy, action, p, q_values, cost, dependent = self.sessions.information_gathering(
            float(parameters.cost_scale),
            float(parameters.cost_slope),
            float(parameters.impatience_fixed),
            float(parameters.impatience_decreasing),
            float(parameters.temperature),
            float(parameters.lapse_rate),
            belief,
            transitions,
            R_CORRECT,
            R_INCORRECT,
            R_NONE,
            data["n_yellow"],
            data["n_seen"],
            data["condition"],
            data["action"],
            self.generate,
            uniforms(trials, self.generate),
        )
        return {
            "policy": policy,
            "action": action,
            "belief": p,
            "q_values": q_values,
            "cost": cost,
            "dependent": dependent,
        }


class _PrepareInformation:
    """The data preparation of `BayesianInformationGathering`."""

    def __init__(self, generate):
        self.generate = generate

    def __call__(self, data):
        out = prepared(data)
        needed = ["n_yellow", "n_seen", "condition"] + ([] if self.generate else ["action"])
        require(out, needed, "BayesianInformationGathering")
        for key in ("n_yellow", "n_seen", "condition"):
            out[key] = indices(out[key])
        out["action"] = indices(out.get("action", numpy.zeros(out["n_seen"].shape[0])))
        return out


class BayesianInformationGathering(Application):
    r"""
    The Bayesian model of information gathering of Hauser et al. (2017a, 2017b), after Moutoussis et al. (2011), for the information sampling task.
    The agent infers which colour forms the majority of the cards from the cards it has opened, and weighs the value of deciding now against the value of opening another card, computed by backward induction, minus a cost of sampling that rises as a sigmoid of the number of cards opened (an urgency signal).

    Parameters
    ----------
    data : pandas.DataFrame
        The data, where each row is one state of a game: the cards opened so far, and what the participant did next. See Notes for the columns required.
    parameters_settings : dict, optional
        The initial values and bounds of the parameters. Each key must be the name of a parameter, and contain a list in the form of [initial, lower_bound, upper_bound]; a single number fixes the parameter at that value instead, so that it is not fitted. Parameters left out keep their default settings. See Notes.
    belief : str
        What the agent infers from the cards it has opened:

            - "majority": the probability that yellow forms the majority of the 25 cards, as in the best-fitting model of Hauser et al. (2017b). Default.
            - "generative": the probability that the process generating the cards produces more yellow than blue cards, as in the model `M_generative` of Hauser et al. (2017b) and in the Bayesian Information Gathering toolbox of the authors.

    generate : bool
        If True, the model plays each game instead of using the observed actions: it opens cards until it decides, and the data must hold all 25 cards of each game. Default is False.

    Returns
    -------
    cpm.generators.Wrapper
        A cpm.generators.Wrapper object.

    Examples
    --------
    >>> from cpm.applications.information_gathering import BayesianInformationGathering
    >>> from cpm.datasets import load_information_sampling_data
    >>> data = load_information_sampling_data()
    >>> data["observed"] = 1
    >>> model = BayesianInformationGathering(data=data[data.ppt == 1])
    >>> model.run()
    >>> model.export()[["policy_0", "policy_1", "policy_2", "action", "dependent"]].head()

    Fit the model to each participant, with the cost scale and slope fixed as in the toolbox of the authors:

    >>> from cpm.optimisation import FminBound, minimise
    >>> model = BayesianInformationGathering(
    ...     data=data[data.ppt == 1],
    ...     parameters_settings={"cost_scale": 10, "cost_slope": 10},
    ... )
    >>> fit = FminBound(
    ...     model=model,
    ...     data=data,
    ...     minimisation=minimise.LogLikelihood.bernoulli,
    ...     ppt_identifier="ppt",
    ...     number_of_starts=5,
    ...     approx_grad=True,
    ... )
    >>> fit.optimise()
    >>> fit.export()

    Notes
    -----

    In the information sampling task (Clark et al., 2006; Hauser et al., 2017a, 2017b), 25 cards are face down on the screen, and the participant has to decide whether the majority of them are yellow or blue. In each game, they open as many cards as they like, one at a time, before they decide; they have to open at least one card. A right answer wins and a wrong answer loses 100 points. In the decreasing condition, the win starts at 250 points and falls by 10 points for every card opened, while a wrong answer still loses 100 points.

    The model implementation uses six parameters:

    - cost_scale: the scale of the cost of sampling, :math:`c`.
    - cost_slope: the slope of the cost of sampling, :math:`k`.
    - impatience_fixed: the number of opened cards at which the cost of sampling reaches half its scale in the fixed condition, the impatience :math:`p_f`.
    - impatience_decreasing: the impatience in the decreasing condition, :math:`p_d`.
    - temperature: the decision temperature, :math:`\tau`. Unlike the inverse temperature of :class:`cpm.models.decision.Softmax`, it divides the values, so that a higher temperature gives more random choices.
    - lapse_rate: the irreducible noise, or lapse rate, :math:`\xi`.

    These are the parameters of the best-fitting model of Hauser et al. (2017b): the scale and slope of the cost are shared across conditions, and the impatience is not. Hauser et al. (2017b) fitted all six; the toolbox of the authors fixes the scale and slope at :math:`c = 10` and :math:`k = 10` by default, which you can do with ``parameters_settings={"cost_scale": 10, "cost_slope": 10}``. The parameters are initialized with the following default values (in the form [initial, lower_bound, upper_bound]), with the bounds of the toolbox of the authors where it fits them:

        - `cost_scale`: [10.0, 0.0, 30.0]
        - `cost_slope`: [10.0, 0.0, 20.0]
        - `impatience_fixed`: [12.5, 0.0, 25.0]
        - `impatience_decreasing`: [12.5, 0.0, 25.0]
        - `temperature`: [5.0, 1.0, 50.0]
        - `lapse_rate`: [0.05, 0.0, 0.5]

    The priors for the parameters are truncated normal distributions, truncated at the bounds, with the following means and standard deviations:

        - `cost_scale`: mean 10.0, standard deviation 5.0.
        - `cost_slope`: mean 10.0, standard deviation 5.0.
        - `impatience_fixed`: mean 12.5, standard deviation 6.25.
        - `impatience_decreasing`: mean 12.5, standard deviation 6.25.
        - `temperature`: mean 5.0, standard deviation 10.0.
        - `lapse_rate`: mean 0.05, standard deviation 0.1.

    Data must contain the following columns, with one row for every state in which the participant acted, that is, after each card they opened:

    - n_seen: the number of cards opened so far, :math:`N`, from 1 to 25.
    - n_yellow: the number of yellow cards among them, :math:`n_y`, from 0 to `n_seen`.
    - condition: the condition of the game, 0 for fixed and 1 for decreasing.
    - action: what the participant did next: 0 for deciding that yellow is the majority, 1 for deciding that blue is, and 2 for opening another card. Not required if `generate` is True.
    - observed: 1 on every row, for the loss function. The dependent variable of the model is the probability of `action`, so that fitting it with :meth:`cpm.optimisation.minimise.LogLikelihood.bernoulli` maximises the log likelihood of the observed actions, :math:`\sum_t \log \pi(a_t \mid n_{y,t}, N_t)`.

    A game in which the participant decided after opening 3 cards therefore has 3 rows, with `action` 2, 2 and then 0 or 1. Yellow and blue are names for the two colours; since the model treats them alike, they can be coded as the colours of the task, or as, say, the colour chosen in the end and the other one, as in :func:`cpm.datasets.load_information_sampling_data`.

    If `generate` is True, the data must hold all 25 cards of each game instead, one row per card, in the order the agent would open them: `n_seen` runs from 1 to 25 in each game, and `n_yellow` counts the yellow cards among them. The agent decides on each row until it chooses a colour; after the last card, it has to choose one. The rows of a game after its decision are not reached, and their `action` is -1. Drop them, with ``export[export.action >= 0]``, to fit the simulated actions.

    .. rubric:: Model Specification

    The model is described in full in the Supplementary Information of Hauser et al. (2017b), and the equation numbers below refer to it. The state of a game is the number of cards opened, :math:`N`, and the number of yellow cards among them, :math:`n_y`. The agent assumes that each card is yellow with an unknown generative probability :math:`q`, with a prior :math:`\text{Beta}(q \mid \alpha, \beta)`, where :math:`\alpha = \beta = 1` (Equation 1.6), so that after :math:`N` cards its belief about :math:`q` is (Equation 1.7):

    .. math::

        P(q \mid n_y, N) = \text{Beta}(q \mid \alpha + n_y, \beta + N - n_y)

    With `belief="majority"`, it infers the probability that yellow forms the majority of the :math:`N_{tot} = 25` cards, which is certain as soon as one colour has more than :math:`N_{tot} / 2` of the opened cards (Equation 1.2). Otherwise, it is the probability that the :math:`N_{tot} - N` cards not yet opened hold enough yellow cards for a majority (Equation 1.3):

    .. math::

        P(MY \mid n_y, N) = \sum_{Y > N_{tot} / 2}^{N_{tot} - (N - n_y)} \int_0^1 \text{Bin}(Y - n_y \mid N_{tot} - N, q) P(q \mid n_y, N) \, dq

    With `belief="generative"`, it infers the probability that the generative probability favours yellow instead:

    .. math::

        P(MY \mid n_y, N) = \int_{1/2}^1 P(q \mid n_y, N) \, dq

    In either case, the probability that blue is the majority is :math:`P(MB \mid n_y, N) = 1 - P(MY \mid n_y, N)`. The value of deciding that yellow is the majority is (Equation 1.8):

    .. math::

        Q(Y \mid n_y, N) = R_{cor} P(MY \mid n_y, N) + R_{inc} P(MB \mid n_y, N)

    and the value of deciding blue, :math:`Q(B \mid n_y, N)`, is calculated analogously, with :math:`R_{cor} = 100` and :math:`R_{inc} = -100`. These rewards are the same in both conditions: the cost of sampling absorbs the falling win of the decreasing condition, as in the best-fitting, `subjective` model of Hauser et al. (2017b). The value of not deciding, :math:`ND`, that is, of opening another card, is the expected value of the next state minus the cost of sampling (Equation 1.10):

    .. math::

        Q(ND \mid n_y, N) = -c_s(N) + \sum_{i \in \{0, 1\}} P(n_y + i, N + 1 \mid n_y, N) V(n_y + i, N + 1)

    where the next card is yellow with the posterior mean of :math:`q` (Equation 1.11):

    .. math::

        P(n_y + 1, N + 1 \mid n_y, N) = \int_0^1 q P(q \mid n_y, N) \, dq = \frac{\alpha + n_y}{\alpha + \beta + N}

    The cost of sampling rises as a sigmoid function of the number of cards opened, :math:`N` (Equation 1.13):

    .. math::

        c_s(N) = \frac{c}{1 + e^{-k (N - p)}}

    where :math:`c` is the `cost_scale`, :math:`k` the `cost_slope`, and the impatience :math:`p` is `impatience_fixed`, :math:`p_f`, in the fixed condition and `impatience_decreasing`, :math:`p_d`, in the decreasing condition. The value of a state is the value of each action weighted by the probability of taking it (Equation 1.9):

    .. math::

        V(n_y, N) = \sum_{a \in \{Y, B, ND\}} \pi(a \mid n_y, N) Q(a \mid n_y, N)

    and the policy is a softmax with the temperature :math:`\tau` and the lapse rate :math:`\xi` (Equation 1.12):

    .. math::

        \pi(a \mid n_y, N) = \frac{e^{Q(a \mid n_y, N) / \tau}}{\sum_{a'} e^{Q(a' \mid n_y, N) / \tau}} (1 - \xi) + \frac{\xi}{3}

    The values are computed by backward induction, from the last card to the first. After the last card, when no card is left to open, not deciding has the value :math:`Q(ND \mid n_y, N_{tot}) = -c_s(N_{tot}) + R_{nd}`, with :math:`R_{nd} = -10000`, as in the toolbox of the authors.

    .. rubric:: Model Output

    For each trial, the model outputs the following variables:

        - `policy`: the probabilities of deciding yellow, deciding blue and opening another card, :math:`\pi(a \mid n_y, N)`.
        - `action`: the observed action, or the action the agent took if `generate` is True, and -1 on the rows of a game after its decision.
        - `belief`: the probability that yellow is the majority, :math:`P(MY \mid n_y, N)`.
        - `q_values`: the values of deciding yellow, deciding blue and opening another card, :math:`Q(a \mid n_y, N)`.
        - `cost`: the cost of opening another card, :math:`c_s(N)`.
        - `dependent`: the probability of `action`, and NaN where `action` is -1.

    .. rubric:: Comparison with the toolbox of the authors

    With `belief="generative"`, the model reproduces the log likelihood that the `Bayesian Information Gathering toolbox <https://github.com/DevComPsy/BayesianInformationGathering>`__ of the authors computes, except that it computes the probability of the next card, Equation 1.11, exactly, where the toolbox integrates it numerically. The difference in the log likelihood is less than 0.001 for the example participants of :func:`cpm.datasets.load_information_sampling_data`. The toolbox also implements forgetting of the evidence and the `objective` costs of Hauser et al. (2017b), which this model does not.

    .. rubric:: Computation

    The policy depends only on the parameters, not on the data, so the model computes it once per run for every state of the game, and only in the conditions present in the data. It computes all trials of a participant in one call, compiled with numba if numba is installed (``pip install cpm-toolbox[numba]``) and as plain Python otherwise, with the same results. `model` is the model function for a single trial, ``model(parameters, trial)``, which returns the outputs of that trial.

    See Also
    --------
    cpm.datasets.load_information_sampling_data : example data of the information sampling task.

    References
    ----------

    Clark, L., Robbins, T. W., Ersche, K. D., & Sahakian, B. J. (2006). Reflection impulsivity in current and former substance users. Biological Psychiatry, 60(5), 515–522.

    Hauser, T. U., Moutoussis, M., Iannaccone, R., Brem, S., Walitza, S., Drechsler, R., Dayan, P., & Dolan, R. J. (2017a). Increased decision thresholds enhance information gathering performance in juvenile Obsessive-Compulsive Disorder (OCD). PLoS Computational Biology, 13(4), e1005440. https://doi.org/10.1371/journal.pcbi.1005440

    Hauser, T. U., Moutoussis, M., NSPN Consortium, Dayan, P., & Dolan, R. J. (2017b). Increased decision thresholds trigger extended information gathering across the compulsivity spectrum. Translational Psychiatry, 7(12), 1296. https://doi.org/10.1038/s41398-017-0040-3

    Moutoussis, M., Bentall, R. P., El-Deredy, W., & Dayan, P. (2011). Bayesian modelling of Jumping-to-Conclusions bias in delusional patients. Cognitive Neuropsychiatry, 16(5), 422–447. https://doi.org/10.1080/13546805.2010.548678

    """

    def __init__(self, data=None, parameters_settings=None, belief="majority", generate=False):
        if belief not in BELIEFS:
            raise ValueError(f"belief must be one of {list(BELIEFS)}, not {belief!r}.")
        self.belief = belief
        parameters = _parameters(parameters_settings)
        self._setup(
            data=data,
            parameters=parameters,
            session_model=_InformationModel(belief, generate),
            prepare=_PrepareInformation(generate),
        )
