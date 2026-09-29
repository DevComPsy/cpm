import numpy as np
import warnings
from cpm.generators import Parameters, Value
from cpm.core import _jit
from cpm.applications._backend import Application, SessionModel, prepared, require, uniforms


def _ptsm_parameters(parameters_settings, utility_curve, weighting):
    """The parameters of `PTSM`."""
    # Use default parameter settings if none provided.
    if parameters_settings is None:
        parameters_settings = {
            "alpha":        [1.0, 1e-2, 5.0],    # alpha: starting value 1.0
            "lambda_loss":  [1.0, 1e-2, 5.0],    # lambda_loss: starting value 1.0
            "gamma":        [0.5, 1e-2, 5.0],    # gamma: starting value 0.5
            "temperature":  [5.0, 1e-2, 15.0]    # temperature: starting value 5.0
        }
        warnings.warn("No parameters specified, using default settings.", stacklevel=3)

    if callable(utility_curve):
        warnings.warn("Utility curve provided, using it instead of power function.", stacklevel=3)
    if utility_curve is not None and not callable(utility_curve):
        raise ValueError("Utility curve must be a callable function.")

    # Create the unified Parameters object with priors.
    params = Parameters(
        alpha=Value(
            value=parameters_settings["alpha"][0],
            lower=parameters_settings["alpha"][1],
            upper=parameters_settings["alpha"][2],
            prior="truncated_normal",
            args={
                "mean": 1.0, "sd": 1.0
            }
        ),
        lambda_loss=Value(
            value=parameters_settings["lambda_loss"][0],
            lower=parameters_settings["lambda_loss"][1],
            upper=parameters_settings["lambda_loss"][2],
            prior="truncated_normal",
            args={"mean": 2.5, "sd": 1.0},
        ),
        gamma=Value(
            value=parameters_settings["gamma"][0],
            lower=parameters_settings["gamma"][1],
            upper=parameters_settings["gamma"][2],
            prior="truncated_normal",
            args={"mean": 2.5, "sd": 1.0},
        ),
        temperature=Value(
            value=parameters_settings["temperature"][0],
            lower=parameters_settings["temperature"][1],
            upper=parameters_settings["temperature"][2],
            prior="truncated_normal",
            args={
                "mean": 10.0, "sd": 5
            }
        ),
        utility_curve=utility_curve,  # Use the piecewise utility transform
        weighting = weighting  # Store the chosen weighting function type
    )
    return params


def _ptsm1992_parameters(parameters_settings, utility_curve, weighting):
    """The parameters of `PTSM1992`."""
    # Use default parameter settings if none provided.
    if parameters_settings is None:
        parameters_settings = {
            "alpha":         [1.0, 0.0, 5.0],   # alpha: starting value 1.0
            "lambda_loss":   [1.0, 0.0, 5.0],   # lambda_loss: starting value 1.0
            "beta":          [1.0, 0.0, 5.0],   # beta: starting value 1.0
            "gamma":         [0.5, 1e-2, 5.0],   # gamma: starting value 0.5
            "delta":         [0.5, 1e-2, 5.0],   # delta: starting value 0.0
            "temperature":   [5.0, 1e-2, 15.0]   # temperature: starting value 5.0
        }
        warnings.warn("No parameters specified, using default settings.", stacklevel=3)

    if callable(utility_curve):
        warnings.warn("Utility curve provided, using it instead of power function.", stacklevel=3)
    if utility_curve is not None and not callable(utility_curve):
        raise ValueError("Utility curve must be a callable function.")


    # Create the unified Parameters object with priors.
    params = Parameters(
        alpha = Value(
            value=parameters_settings["alpha"][0],
            lower=parameters_settings["alpha"][1],
            upper=parameters_settings["alpha"][2],
            prior="truncated_normal",
            args={
                "mean": 2.5, "sd": 1.0
            }
        ),
        lambda_loss=Value(
            value=parameters_settings["lambda_loss"][0],
            lower=parameters_settings["lambda_loss"][1],
            upper=parameters_settings["lambda_loss"][2],
            prior="truncated_normal",
            args={"mean": 2.5, "sd": 1.0},
        ),
        beta=Value(
            value=parameters_settings["beta"][0],
            lower=parameters_settings["beta"][1],
            upper=parameters_settings["beta"][2],
            prior="truncated_normal",
            args={"mean": 2.5, "sd": 1.0},
        ),
        gamma=Value(
            value=parameters_settings["gamma"][0],
            lower=parameters_settings["gamma"][1],
            upper=parameters_settings["gamma"][2],
            prior="truncated_normal",
            args={"mean": 2.5, "sd": 1.0},
        ),
        delta=Value(
            value=parameters_settings["delta"][0],
            lower=parameters_settings["delta"][1],
            upper=parameters_settings["delta"][2],
            prior="truncated_normal",
            args={"mean": 2.5, "sd": 1.0},
        ),
        temperature=Value(
            value=parameters_settings["temperature"][0],
            lower=parameters_settings["temperature"][1],
            upper=parameters_settings["temperature"][2],
            prior="truncated_normal",
            args={"mean": 10, "sd": 2.5},
        ),
        utility_curve=utility_curve,  # Use the piecewise utility transform
        weighting = weighting  # Store the chosen weighting function type
    )
    return params


def _ptsm2025_transform(x, alpha):
    ## Piecewise utility transform, as the model computes it
    return _jit.kernels("cpm.applications._sessions", python=True).power_utility(x, alpha)


def _ptsm2025_parameters(parameters_settings, utility_curve, variant):
    """The parameters of `PTSM2025`."""
    if parameters_settings is None:
        warnings.warn("No parameters specified, using JAGS-inspired defaults.", stacklevel=3)
        parameters_settings = {
            "eta":         [0.0,   -0.49,  0.49],
            "phi_gain":    [0.0,   -10.0,  10.0],
            "phi_loss":    [0.0,   -10.0,  10.0],
            "temperature": [5.0,    0.001, 20.0],
            "alpha":       [1.0,    0.001,  5.0],
        }

    if callable(utility_curve):
        warnings.warn("Utility curve provided, using it instead of power function.", stacklevel=3)
    if utility_curve is not None and not callable(utility_curve):
        raise ValueError("Utility curve must be a callable function.")

    parameters = Parameters(
        eta=Value(
            value=parameters_settings["eta"][0],
            lower=parameters_settings["eta"][1],
            upper=parameters_settings["eta"][2],
            prior="truncated_normal",
            args={"mean": 0.0, "sd": 0.25}
        ),
        phi_gain=Value(
            value=parameters_settings["phi_gain"][0],
            lower=parameters_settings["phi_gain"][1],
            upper=parameters_settings["phi_gain"][2],
            prior="truncated_normal",
            args={"mean": 0.0, "sd": 2.5}
        ),
        phi_loss=Value(
            value=parameters_settings["phi_loss"][0],
            lower=parameters_settings["phi_loss"][1],
            upper=parameters_settings["phi_loss"][2],
            prior="truncated_normal",
            args={"mean": 0.0, "sd": 2.5}
        ),
        temperature=Value(
            value=parameters_settings["temperature"][0],
            lower=parameters_settings["temperature"][1],
            upper=parameters_settings["temperature"][2],
            prior="truncated_normal",
            args={
                "mean": 10.0, "sd": 5
            }
        ),
        utility_curvature=_ptsm2025_transform,
    )

    if variant == "alpha":
        parameters["alpha"] = Value(
            value=parameters_settings["alpha"][0],
            lower=parameters_settings["alpha"][1],
            upper=parameters_settings["alpha"][2],
            prior="truncated_normal",
            args={
                "mean": 1.0, "sd": 1.0
            }
        )
    else:
        parameters["alpha"] = 1.0
    return parameters



def _weighting_code(weighting):
    """The number of a weighting function in `cpm.models.kernels.WEIGHTING`."""
    from cpm.models.kernels import WEIGHTING

    if weighting not in WEIGHTING:
        raise ValueError(
            "Invalid weighting type. Must be one of: 'tk', 'power', 'prelec', 'gw'."
        )
    return WEIGHTING[weighting]


class _PrepareRisky:
    """The data preparation of the prospect-theory session models."""

    def __init__(self, columns, model):
        self.columns = columns
        self.model = model

    def __call__(self, data):
        out = prepared(data)
        require(out, self.columns + ["observed"], self.model)
        for key in self.columns:
            out[key] = np.ascontiguousarray(out[key], dtype=np.float64)
        out["observed"] = np.ascontiguousarray(out["observed"], dtype=np.int64)
        return out


RISKY_COLUMNS = ["safe_magnitudes", "risky_magnitudes", "risky_probability"]



class _ProspectModel(SessionModel):
    """The model function of `PTSM` and `PTSM1992`, for all trials at once."""

    def __init__(self, weighting, generate, choose_always, dependent_chosen, separate):
        super().__init__(generate)
        self.weighting = _weighting_code(weighting)
        self.choose_always = choose_always
        self.dependent_chosen = dependent_chosen
        ## whether losses have their own curvatures (beta, delta), as in PTSM1992
        self.separate = separate

    def utilities(self, parameters, data, alpha, lambda_loss):
        """The utilities of a user-supplied utility curve, called as `ProspectUtility` calls it."""
        curve = parameters.utility_curve if self.separate else None
        trials = data["observed"].shape[0]
        if curve is None:
            return False, np.empty((0, 2))
        out = np.empty((trials, 2))
        for t in range(trials):
            for j, key in enumerate(("safe_magnitudes", "risky_magnitudes")):
                x = np.array([data[key][t]], dtype=float)
                out[t, j] = np.sum(curve(x=x, alpha=alpha, lambda_loss=lambda_loss))
        return True, out

    def __call__(self, parameters, data):
        alpha = float(parameters.alpha)
        lambda_loss = float(parameters.lambda_loss)
        gamma = float(parameters.gamma)
        if self.separate:
            beta, delta = float(parameters.beta), float(parameters.delta)
        else:
            beta, delta = alpha, 1.0
        custom, utilities = self.utilities(parameters, data, alpha, lambda_loss)
        observed = data["observed"]
        trials = observed.shape[0]
        out = self.sessions.prospect_softmax(
            alpha, beta, lambda_loss, gamma, delta, float(parameters.temperature),
            data["safe_magnitudes"], data["risky_magnitudes"], data["risky_probability"],
            observed, self.weighting, utilities, custom, self.choose_always, self.generate,
            uniforms(trials, self.choose_always or self.generate), self.dependent_chosen,
        )
        policy, dependent, chosen, optimal, best, ev_risk, u_safe, u_risk = out
        return {
            "policy": policy,
            "dependent": dependent,
            "observed": observed,
            "chosen": chosen,
            "is_optimal": optimal,
            "objective_best": best,
            "ev_safe": data["safe_magnitudes"],
            "ev_risk": ev_risk,
            "u_safe": u_safe,
            "u_risk": u_risk,
        }


class _PTSM2025Model(SessionModel):
    """The model function of `PTSM2025`, for all trials at once."""

    def __call__(self, parameters, data):
        observed = data["observed"]
        policy, choice, u_safe, u_risk = self.sessions.ptsm2025(
            float(parameters.eta),
            float(parameters.phi_gain),
            float(parameters.phi_loss),
            float(parameters.temperature),
            float(parameters.alpha),
            data["safe_magnitudes"],
            data["risky_magnitudes"],
            data["risky_probability"],
            data["ambiguity"],
            uniforms(observed.shape[0], True),
        )
        return {
            "policy": policy,
            "model_choice": choice,
            "real_choice": observed,
            "u_safe": u_safe,
            "u_risk": u_risk,
            "dependent": policy,
        }


class PTSM(Application):
    r"""
    A simplified version of the Prospect Theory-based Softmax Model (PTSM) for decision-making tasks based on Tversky & Kahneman (1992), similar to the initial publication of the theory in Kahneman & Tversky (1979). It differs from :class:`cpm.applications.decision_making.PTSM2025` and :class:`cpm.applications.decision_making.PTSM1992` in that it does not use use different utility and weight curvature parameters for gains and losses.
    
    Parameters
    ----------
    data : pd.DataFrame
        The data, where each row is a trial and each column is an input to the model. Expected to have columns: 'safe_magnitudes', 'risky_magnitudes', 'risky_probability', 'observed'.
    parameters_settings : dict, optional
        A dictionary containing the initial values and bounds for the model parameters. Each key must correspond to the name of the parameter, and contain a list in the form of [initial, lower_bound, upper_bound]. If not provided, default values are used. See Notes.
    utility_curve : callable, optional
        A callable function that defines the utility curve. If provided, it overrides the default power function used for utility transformations. Its first argument should be the magnitude, and the second argument should be the curvature parameter (alpha). If None, a power function is used, see Notes.
    weighting : str
        The probability weighting function to use. Options include:

            - "power": use a simple power function (p^gamma)
            - "tk": use the Tversky–Kahneman (1992) weighting function.

        See :class:`cpm.models.activation.ProspectUtility` for explanation and alternatives.

    Returns
    -------
    cpm.generators.Wrapper
        An instance of the PTSM model, which can be used to fit data and generate predictions.

    Notes
    -----

    The model parameters are initialized with the following default values if not specified (values are in the form [initial, lower_bound, upper_bound]):

        - `alpha`: [1.0, 1e-2, 5.0] (utility curvature for both gains and losses)
        - `lambda_loss`: [1.0, 1e-2, 5.0] (loss sensitivity)
        - `gamma`: [0.5, 1e-2, 5.0] (curvature for the weighting function for both gains and losses)
        - `temperature`: [5.0, 1e-2, 15.0] (temperature parameter for softmax)

    The priors for the parameters are set as follows:

        - `alpha`: truncated normal with mean 1.0 and standard deviation 1.0.
        - `lambda_loss`: truncated normal with mean 2.5 and standard deviation 1.0.
        - `gamma`: truncated normal with mean 2.5 and standard deviation 1.0.
        - `temperature`: truncated normal with mean 10.0 and standard deviation 5.0.

    .. rubric:: Model Specification

    The model computes the subjective utility of the safe and risky options using a utility function, which can be either a power function or a user-defined utility curve. If a utility curve is not provided, the model uses the following power function with curvature parameter :math:`\alpha` after Tversky & Kahneman (1992):
    
    .. math::

        \mathcal{U}(o) = \sum_{i=1}^{n} w(p_i) \cdot u(x_i)

    where :math:`w` is a weighting function of the probability p of a potential outcome,
    and :math:`u` is the utility function of the magnitude x of a potential outcome. The choice options is denoted with :math:`o`.
    The utility function :math:`u` is defined as a power function for both gains and losses. It is implemented
    after Equation 5 in Tversky & Kahneman (1992):

    .. math::

        u(x) =
        \begin{cases}
        x^\alpha & \text{if } x \geq 0 \\
        -\lambda \cdot (-x)^\alpha & \text{if } x < 0
        \end{cases}

    where :math:`\alpha` is the utility curvature parameter for both gains and losses, and :math:`\lambda` is the loss aversion parameter.
    The weighting function is implemented after Equation 6 in Tversky & Kahneman (1992):

    .. math::

        w(p) = \frac{p^\gamma}{(p^\gamma + (1 - p)^\gamma)^{1/\gamma}}

    where `gamma`, denoted via :math:`\gamma`, is the discriminability parameter of the weighting function for both gains and losses.
    The model then applies the softmax function to compute the choice probabilities:

    .. math::

        p(o_i) = \frac{e^{\beta \cdot \mathcal{U}(o_i)}}{\sum_{j=1}^{n} e^{\beta \cdot \mathcal{U}(o_j)}}

    .. rubric:: Model output

    The model outputs the following trial-level information:

        - `policy`: the softmax probabilities for each option.
        - `dependent`: the probability of choosing the risky option.
        - `observed`: the observed (participant's) choice (0 for safe, 1 for risky).
        - `chosen`: the chosen option based on the softmax probabilities.
        - `is_optimal`: whether the chosen option is optimal (1 if chosen option is objectively better, 0 otherwise).
        - `objective_best`: the objectively better option (1 for risky, 0 for safe) determined by the objective evidence for each.
        - `ev_safe`: the expected value of the safe option.
        - `ev_risk`: the expected value of the risky option.
        - `u_safe`: the utility of the safe option.
        - `u_risk`: the utility of the risky option.
    
    .. rubric:: Details for fitting the model to data

    The model uses a softmax function to map the computed utilities to choice probabilities, with a temperature parameter that controls the stochasticity of the choices. Exponential functions, depending on the temperature parameter, can get out of hand quickly, so it is advisable to keep the temperature parameter within reasonable bounds (e.g., between 0.001 and 20.0).

    If you get **overflow warnings** during fitting, consider lowering the upper bound of the temperature parameter. Another possible reason for these **overflow warnings** is that the computed utilities are very large in magnitude. Ensure that the magnitudes in your dataset are within a reasonable range (e.g., between 0 and 1, or -1 and 1). Another option is to z-score the utilities before passing them to the softmax function, which can help stabilize the exponentials. 

    .. rubric:: Computation

    The model computes all trials of a participant in one call, compiled with numba if numba is installed (``pip install cpm-toolbox[numba]``) and as plain Python otherwise, with the same results. `model` is the model function for a single trial, ``model(parameters, trial)``, which returns the outputs of that trial.

    See Also
    --------
    cpm.models.decision.Softmax : for mapping utilities to choice probabilities.

    cpm.models.activation.ProspectUtility : for the Prospect Utility class that computes subjective utilities and weighted probabilities.

    References
    ----------

    Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2), 263–291.

    Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. Journal of Risk and uncertainty, 5, 297-323.
    
    """

    def __init__(
        self,
        data=None,
        parameters_settings=None,
        generate=False,
        utility_curve=None,  # Callable function for utility transformation
        weighting="tk"  # Options: "tk" or "power"
    ):
        params = _ptsm_parameters(parameters_settings, utility_curve, weighting)
        self._setup(
            data=data,
            parameters=params,
            session_model=_ProspectModel(weighting, generate, choose_always=False,
                                         dependent_chosen=True, separate=False),
            prepare=_PrepareRisky(RISKY_COLUMNS, "PTSM"),
        )


class PTSM1992(Application):
    r"""
    A Prospect Theory-based Softmax Model (PTSM) for decision-making tasks based on Tversky & Kahneman (1992), similar to the initial publication of the theory in Kahneman & Tversky (1979). It computes expected utility by combining transformed magnitudes and weighted probabilities, suitable for safe–risky decision paradigms.

    The model computes objective EV internally (ev_safe vs. ev_risk)
    and outputs trial-level information (including whether the chosen option is optimal).
    
    Additionally, the model accepts a "weighting" argument that determines 
    which probability weighting function to use when computing the subjective 
    weighting of risky probabilities.

    Parameters
    ----------
    data : pd.DataFrame
        The data, where each row is a trial and each column is an input to the model. Expected to have columns: 'safe_magnitudes', 'risky_magnitudes', 'risky_probability', 'observed'.
    parameters_settings : dict, optional
        A dictionary containing the initial values and bounds for the model parameters. Each key must correspond to the name of the parameter, and contain a list in the form of [initial, lower_bound, upper_bound]. If not provided, default values are used. See Notes.
    utility_curve : callable, optional
        A callable function that defines the utility curve. If provided, it overrides the default power function used for utility transformations. Its first argument should be the magnitude. The following variables are also passed to this function: `alpha`, `beta` and `lambda_loss`. If None, a power function is used, see Notes.
    weighting : str
        The probability weighting function to use. Options include:

            - "power": use a simple power function (p^gamma)
            - "tk": use the Tversky–Kahneman (1992) weighting function.

        See :class:`cpm.models.activation.ProspectUtility` for explanation and alternatives.

    Returns
    -------
    cpm.generators.Wrapper
        An instance of the PTSM1992 model, which can be used to fit data and generate predictions.


    Notes
    -----

    The model parameters are initialized with the following default values if not specified (values are in the form [initial, lower_bound, upper_bound]):

        - `alpha`: [1.0, 0, 5.0] (utility curvature for gains)
        - `beta`: [1.0, 0, 5.0] (utility curvature for losses)
        - `lambda_loss`: [1.0, 0, 5.0] (loss sensitivity)
        - `gamma`: [0.5, 0.001, 5.0] (curvature for gains)
        - `delta`: [0.5, 0.001, 5.0] (curvature for losses)
        - `temperature`: [5.0, 0.001, 20.0] (temperature parameter for softmax)

    The priors for the parameters are set as follows:

        - `alpha`: truncated normal with mean 2.5 and standard deviation 1.0.
        - `beta`: truncated normal with mean 2.5 and standard deviation 1.0.
        - `lambda_loss`: truncated normal with mean 2.5 and standard deviation 1.0.
        - `gamma`: truncated normal with mean 2.5 and standard deviation 1.0.
        - `delta`: truncated normal with mean 0 and standard deviation 1.0.
        - `temperature`: truncated normal with mean 10 and standard deviation 2.5.


    .. rubric:: Model Specification

    The model computes the subjective utility of the safe and risky options using a utility function, which can be either a power function or a user-defined utility curve. If a utility curve is not provided, the model uses the following power function with curvature parameter :math:`\alpha` after Tversky & Kahneman (1992):
    
    .. math::

        \mathcal{U}(o) = \sum_{i=1}^{n} w(p_i) \cdot u(x_i)

    where :math:`w` is a weighting function of the probability p of a potential outcome,
    and :math:`u` is the utility function of the magnitude x of a potential outcome. The choice options is denoted with :math:`o`.
    The utility function :math:`u` is defined as a power function for both gains and losses. It is implemented
    after Equation 5 in Tversky & Kahneman (1992):

    .. math::

        u(x) =
        \begin{cases}
        x^\alpha & \text{if } x \geq 0 \\
        -\lambda \cdot (-x)^\beta & \text{if } x < 0
        \end{cases}

    where :math:`\alpha` is the utility curvature parameter for gains, and :math:`\beta`, is the curvature parameter for losses, :math:`\lambda` is the loss aversion parameter.
    The weighting function is implemented after Equation 6 in Tversky & Kahneman (1992):

    .. math::

        w^{+}(p) = \frac{p^\gamma}{(p^\gamma + (1 - p)^\gamma)^{1/\gamma}}, w^{-}(p) = \frac{p^\delta}{(p^\delta + (1 - p)^\delta)^{1/\delta}}

    where `gamma`, denoted via :math:`\gamma`, is the discriminability parameter of the weighting function for gains, and with `delta`, denoted via :math:`\delta`, is the discriminability parameter of the weighting function for losses.

    The model then applies the softmax function to compute the choice probabilities:

    .. math::

        p(o_i) = \frac{e^{beta \cdot \mathcal{U}(o_i)}}{\sum_{j=1}^{n} e^{beta \cdot \mathcal{U}(o_j)}}

    .. rubric:: Model output

    The model outputs the following trial-level information:

        - `policy`: the softmax probabilities for each option.
        - `dependent`: the probability of choosing the risky option.
        - `observed`: the observed (participant's) choice (0 for safe, 1 for risky).
        - `chosen`: the chosen option based on the softmax probabilities.
        - `is_optimal`: whether the chosen option is optimal (1 if chosen option is objectively better, 0 otherwise).
        - `objective_best`: the objectively better option (1 for risky, 0 for safe) determined by the objective evidence for each.
        - `ev_safe`: the expected value of the safe option.
        - `ev_risk`: the expected value of the risky option.
        - `u_safe`: the utility of the safe option.
        - `u_risk`: the utility of the risky option.

    .. rubric:: Details for fitting the model to data

    The model uses a softmax function to map the computed utilities to choice probabilities, with a temperature parameter that controls the stochasticity of the choices. Exponential functions, depending on the temperature parameter, can get out of hand quickly, so it is advisable to keep the temperature parameter within reasonable bounds (e.g., between 0.001 and 20.0).

    If you get **overflow warnings** during fitting, consider lowering the upper bound of the temperature parameter. Another possible reason for these **overflow warnings** is that the computed utilities are very large in magnitude. Ensure that the magnitudes in your dataset are within a reasonable range (e.g., between 0 and 1, or -1 and 1). Another option is to z-score the utilities before passing them to the softmax function, which can help stabilize the exponentials. 
        

    .. rubric:: Computation

    The model computes all trials of a participant in one call, compiled with numba if numba is installed (``pip install cpm-toolbox[numba]``) and as plain Python otherwise, with the same results. `model` is the model function for a single trial, ``model(parameters, trial)``, which returns the outputs of that trial.

    See Also
    --------
    cpm.models.decision.Softmax : for mapping utilities to choice probabilities.

    cpm.models.activation.ProspectUtility : for the Prospect Utility class that computes subjective utilities and weighted probabilities.

    References
    ----------

    Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2), 263–291.

    Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. Journal of Risk and uncertainty, 5, 297-323.

    """

    def __init__(
        self,
        data=None,
        parameters_settings=None,
        utility_curve=None,
        weighting="tk"  # Options: "tk" or "power"
    ):
        params = _ptsm1992_parameters(parameters_settings, utility_curve, weighting)
        self._setup(
            data=data,
            parameters=params,
            session_model=_ProspectModel(weighting, generate=False, choose_always=True,
                                         dependent_chosen=False, separate=True),
            prepare=_PrepareRisky(RISKY_COLUMNS, "PTSM1992"),
        )


class PTSM2025(Application):
    r"""
    An Prospect Theory Softmax Model loosely based on Chew et al. (2019), incorporating a bias term (phi_gain / phi_loss) in the softmax function for risks and gains, a utility curvature parameter (alpha) for non-linear utility transformations, and an ambiguity aversion parameter (eta).

    Parameters
    ----------
    data : pd.DataFrame, optional
        Data containing the trials to be modeled, where each row represents a trial in the experiment (a state), and each column represents a variable (e.g., safe_magnitudes, risky_magnitudes, risky_probability, ambiguity, observed variable).
    parameters_settings : dict, optional
        A dictionary containing the initial values and bounds for the model parameters. Each key must correspond to the name of the parameter, and contain a list in the form of [initial, lower_bound, upper_bound]. If not provided, default values are used. See Notes.
    utility_curve : callable, optional
        A callable function that defines the utility curve. If provided, it overrides the default power function used for utility transformations. Its first argument should be the magnitude, and the second argument should be the curvature parameter (alpha). If None, a power function is used, see Notes.
    variant : str, optional
        The variant of the model to use. Options are "alpha" for the full model with a non-linear curvature or "standard" for a simplified version without curvature. Default is "alpha".

    Returns
    -------
    cpm.generators.Wrapper
        An instance of the PTSM2025 model, which can be used to fit data and generate predictions.

    Notes
    -----
    The model parameters are initialized with the following default values if not specified (values are in the form [initial, lower_bound, upper_bound]):

        - `eta`: [0.0, -0.49, 0.49] (ambiguity aversion)
        - `phi_gain`: [0.0, -10.0, 10.0] (gain sensitivity)
        - `phi_loss`: [0.0, -10.0, 10.0] (loss sensitivity)
        - `temperature`: [5.0, 0.001, 20.0] (temperature parameter)
        - `alpha`: [1.0, 0.001, 5.0] (utility curvature parameter)

    The priors for the parameters are set as follows:

        - `eta`: truncated normal with mean 0.0 and standard deviation 0.25.
        - `phi_gain`: truncated normal with mean 0.0 and standard deviation 2.5.
        - `phi_loss`: truncated normal with mean 0.0 and standard deviation 2.5.
        - `temperature`: truncated normal with mean 10.0 and standard deviation 5.
        - `alpha`: truncated normal with mean 1.0 and standard deviation 1.

    .. rubric:: Model Description

    In what follows, we briefly describe the model's operations. First, the model calculates the subjective probability of the risky option, adjusting for ambiguity aversion using the parameter `eta`, denoted with :math:`\eta`. The subjective probability is computed as:

    .. math::

        p_{subjective} = p_{risky} - \eta \cdot ambiguity

    where :math:`p_{risky}` is the original probability of the risky choice and :math:`ambiguity` is the ambiguity associated with the risky option, either 0 for non-ambiguous or 1 for ambiguous cases.
    The utility of the safe and risky options is then computed using a utility function, which can be either a power function or a user-defined utility curve.
    If a utility curve is not provided, the model uses the following power function with curvature parameter `alpha`, denoted with :math:`\alpha`:

    .. math::

        u(x) =
        \begin{cases}
        x^\alpha & \text{if } x \geq 0 \\
        -|x|^\alpha & \text{if } x < 0
        \end{cases}

    The model then applies loss aversion and gain sensitivity adjustments based on the sign of the risky choice magnitude. Here, the gain sensitivity `phi_gain`, denoted as :math:`\phi_{gain}`, is applied when the risky choice is positive, and the loss sensitivity `phi_loss`, denoted as :math:`\phi_{loss}`, is applied when the risky choice is negative. The adjusted probability of choosing the risky option, :math:`p(A_{risky})`, is computed using a softmax function:

    .. math::

        p(A_{risky}) = \frac{e^{\beta (u_{risky} + \phi_{t})}}{e^{\beta (u_{risky} + \phi_{t})} + e^{\beta u_{safe}}}

    where denoted with :math:`\beta` is the `temperature` parameter, :math:`u_{risky}` is the utility of the risky option, :math:`u_{safe}` is the utility of the safe option, and :math:`\phi_{t}` is either :math:`\phi_{gain}` or :math:`\phi_{loss}` depending on the sign of the risky choice magnitude. Note that in Chew et al. (2019), the model only has a gambling bias term for the gain loss, that is then added to the difference between the safe and risky utilities, and only then transformed to a probability via a sigmoid function.

    Furthermore, the model generates a response based on the computed probabilities, where the choice is sampled from a Bernoulli distribution with the computed policy as the probability of choosing the risky option.

    .. rubric:: Model Output

    For each trial, the model outputs the following variables:

        - `policy`: The computed probabilities for the risky options.
        - `model_choice`: The model's predicted choice (0 for safe, 1 for risky).
        - `real_choice`: The observed (participant's) choice from the data.
        - `u_safe`: The utility of the safe option.
        - `u_risk`: The utility of the risky option.
        - `dependent`: The computed probability of a risky choice according to the model, which can be used for further analysis or fitting
    

    .. rubric:: Details for fitting the model to data

    The model uses a softmax function to map the computed utilities to choice probabilities, with a temperature parameter that controls the stochasticity of the choices. Exponential functions, depending on the temperature parameter, can get out of hand quickly, so it is advisable to keep the temperature parameter within reasonable bounds (e.g., between 0.001 and 20.0).

    If you get **overflow warnings** during fitting, consider lowering the upper bound of the temperature parameter. Another possible reason for these **overflow warnings** is that the computed utilities are very large in magnitude. Ensure that the magnitudes in your dataset are within a reasonable range (e.g., between 0 and 1, or -1 and 1). Another option is to z-score the utilities before passing them to the softmax function, which can help stabilize the exponentials. 

    .. rubric:: Computation

    The model computes all trials of a participant in one call, compiled with numba if numba is installed (``pip install cpm-toolbox[numba]``) and as plain Python otherwise, with the same results. `model` is the model function for a single trial, ``model(parameters, trial)``, which returns the outputs of that trial.

    References
    ----------
    Chew, B., Hauser, T. U., Papoutsi, M., Magerkurth, J., Dolan, R. J., & Rutledge, R. B. (2019). Endogenous fluctuations in the dopaminergic midbrain drive behavioral choice variability. Proceedings of the National Academy of Sciences, 116(37), 18732–18737. https://doi.org/10.1073/pnas.1900872116
    """

    def __init__(
        self,
        data=None,
        parameters_settings=None,
        utility_curve=None,
        variant="alpha"
    ):
        self.variant = variant
        parameters = _ptsm2025_parameters(parameters_settings, utility_curve, variant)
        self._setup(
            data=data,
            parameters=parameters,
            session_model=_PTSM2025Model(),
            prepare=_PrepareRisky(RISKY_COLUMNS + ["ambiguity"], "PTSM2025"),
        )
