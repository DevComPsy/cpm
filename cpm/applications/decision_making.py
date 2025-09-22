import copy
import numpy as np
import warnings
from cpm.generators import Wrapper, Parameters, Value
from cpm.models.decision import Softmax
from cpm.models.activation import ProspectUtility


class PTSM(Wrapper):
    """
    A simplified version of the Prospect Theory-based Softmax Model (PTSM) for decision-making tasks based on Tversky & Kahneman (1992), similar to the initial publication of the theory in Kahneman & Tversky (1979). It differs from [cpm.applications.decision_making.PTSM2025][cpm.applications.decision_making.PTSM2025] and [cpm.applications.decision_making.PTSM1992][cpm.applications.decision_making.PTSM1992] in that it does not use use different utility and weight curvature parameters for gains and losses.
    
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

        See [cpm.models.activation.ProspectUtility][cpm.models.activation.ProspectUtility] for explanation and alternatives.

    Returns
    -------
    cpm.generators.Wrapper
        An instance of the PTSM model, which can be used to fit data and generate predictions.

    Notes
    ------

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

    ### Model Specification

    The model computes the subjective utility of the safe and risky options using a utility function, which can be either a power function or a user-defined utility curve. If a utility curve is not provided, the model uses the following power function with curvature parameter $\\alpha$ after Tversky & Kahneman (1992):
    
    $$
    \\mathcal{U}(o) = \\sum_{i=1}^{n} w(p_i) \\cdot u(x_i)
    $$

    where $w$ is a weighting function of the probability p of a potential outcome,
    and $u$ is the utility function of the magnitude x of a potential outcome. The choice options is denoted with $o$.
    The utility function $u$ is defined as a power function for both gains and losses. It is implemented
    after Equation 5 in Tversky & Kahneman (1992):

    $$
    u(x) =
    \\begin{cases}
        x^\\alpha & \\text{if } x \\geq 0 \\\\
        -\\lambda \\cdot (-x)^\\alpha & \\text{if } x < 0
    \\end{cases}
    $$

    where $\\alpha$ is the utility curvature parameter for both gains and losses, and $\\lambda$ is the loss aversion parameter.
    The weighting function is implemented after Equation 6 in Tversky & Kahneman (1992):

    $$
    w(p) = \\frac{p^\\gamma}{(p^\\gamma + (1 - p)^\\gamma)^{1/\\gamma}}
    $$

    where `gamma`, denoted via $\\gamma$, is the discriminability parameter of the weighting function for both gains and losses.
    The model then applies the softmax function to compute the choice probabilities:

    $$
    p(o_i) = \\frac{e^{\\beta \\cdot \\mathcal{U}(o_i)}}{\\sum_{j=1}^{n} e^{\\beta \\cdot \\mathcal{U}(o_j)}}
    $$
    
    ### Model output

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
    
        

    See Also
    ---------
    [cpm.models.decision.Softmax][cpm.models.decision.Softmax] : for mapping utilities to choice probabilities.

    [cpm.models.activation.ProspectUtility][cpm.models.activation.ProspectUtility] : for the Prospect Utility class that computes subjective utilities and weighted probabilities.

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
        # Use default parameter settings if none provided.
        if parameters_settings is None:
            parameters_settings = {
                "alpha":        [1.0, 1e-2, 5.0],    # alpha: starting value 1.0
                "lambda_loss":  [1.0, 1e-2, 5.0],    # lambda_loss: starting value 1.0
                "gamma":        [0.5, 1e-2, 5.0],    # gamma: starting value 0.5
                "temperature":  [5.0, 1e-2, 15.0]    # temperature: starting value 5.0
            }
            warnings.warn("No parameters specified, using default settings.")

        if callable(utility_curve):
            warnings.warn("Utility curve provided, using it instead of power function.")
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

        def model_fn(parameters, trial):
            """
            Called per trial. Computes the subjective utility for two options based on prospect theory,
            using an external weighting function from the ProspectUtility class.
            """
            # Extract parameter values
            alpha = copy.deepcopy(parameters.alpha.value)
            lambd = copy.deepcopy(parameters.lambda_loss.value)
            gamma = copy.deepcopy(parameters.gamma.value)  
            temp  = copy.deepcopy(parameters.temperature.value)

            # Read trial data 
            safe_magn  = trial["safe_magnitudes"]
            risky_magn = trial["risky_magnitudes"]
            risky_prob = trial["risky_probability"]
            observed = trial["observed"].astype(int)

            # Compute objective expected values (EV)
            ev_safe = safe_magn
            ev_risk = risky_magn * risky_prob

            # Determine which option is objectively better
            objective_best = 1 if ev_risk >= ev_safe else 0

            pt_util = ProspectUtility(
                magnitudes=np.array([safe_magn, risky_magn]),
                probabilities=np.array([1.0, risky_prob]),
                alpha=alpha,
                lambda_loss=lambd,
                gamma=gamma,
                weighting=parameters.weighting
            )
            subjective_utilities = pt_util.compute()


            # Compute softmax probabilities using the specified temperature
            sm = Softmax(temperature=temp, activations=subjective_utilities)
            policies = sm.compute()
            prob_chosen = policies[observed]

            # Determine choice: generate a response if required, else use the observed one
            chosen = sm.choice() if generate else observed
            # Determine if the chosen option is optimal
            is_optimal = 1 if chosen == objective_best else 0


            return {
                "policy": policies,
                "dependent": np.array([prob_chosen]),
                "observed": observed,  # Ensure the optimizer sees the 'observed' column
                "chosen": chosen,
                "is_optimal": is_optimal,
                "objective_best": objective_best,
                "ev_safe": ev_safe,
                "ev_risk": ev_risk,
                "u_safe": subjective_utilities[0],
                "u_risk": subjective_utilities[1],
            }

        # Pass the model function and parameters to the parent Wrapper
        super().__init__(data=data, model=model_fn, parameters=params)


class PTSM1992(Wrapper):
    """
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

        See [cpm.models.activation.ProspectUtility][cpm.models.activation.ProspectUtility] for explanation and alternatives.

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


    ### Model Specification

    The model computes the subjective utility of the safe and risky options using a utility function, which can be either a power function or a user-defined utility curve. If a utility curve is not provided, the model uses the following power function with curvature parameter $\\alpha$ after Tversky & Kahneman (1992):
    
    $$
    \\mathcal{U}(o) = \\sum_{i=1}^{n} w(p_i) \\cdot u(x_i)
    $$

    where $w$ is a weighting function of the probability p of a potential outcome,
    and $u$ is the utility function of the magnitude x of a potential outcome. The choice options is denoted with $o$.
    The utility function $u$ is defined as a power function for both gains and losses. It is implemented
    after Equation 5 in Tversky & Kahneman (1992):

    $$
    u(x) =
    \\begin{cases}
        x^\\alpha & \\text{if } x \\geq 0 \\\\
        -\\lambda \\cdot (-x)^\\beta & \\text{if } x < 0
    \\end{cases}
    $$

    where $\\alpha$ is the utility curvature parameter for gains, and $\\beta$, is the curvature parameter for losses, $\\lambda$ is the loss aversion parameter.
    The weighting function is implemented after Equation 6 in Tversky & Kahneman (1992):

    $$
    w^{+}(p) = \\frac{p^\\gamma}{(p^\\gamma + (1 - p)^\\gamma)^{1/\\gamma}}, w^{-}(p) = \\frac{p^\\delta}{(p^\\delta + (1 - p)^\\delta)^{1/\\delta}}
    $$

    where `gamma`, denoted via $\\gamma$, is the discriminability parameter of the weighting function for gains, and with `delta`, denoted via $\\delta$, is the discriminability parameter of the weighting function for losses.

    The model then applies the softmax function to compute the choice probabilities:

    $$
    p(o_i) = \\frac{e^{beta \\cdot \\mathcal{U}(o_i)}}{\\sum_{j=1}^{n} e^{beta \\cdot \\mathcal{U}(o_j)}}
    $$

    ### Model output

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
    
        

    See Also
    ---------
    [cpm.models.decision.Softmax][cpm.models.decision.Softmax] : for mapping utilities to choice probabilities.

    [cpm.models.activation.ProspectUtility][cpm.models.activation.ProspectUtility] : for the Prospect Utility class that computes subjective utilities and weighted probabilities.

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
            warnings.warn("No parameters specified, using default settings.")

        if callable(utility_curve):
            warnings.warn("Utility curve provided, using it instead of power function.")
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

        def model_fn(parameters, trial):
            """
            Called per trial. Computes the subjective utility for two options based on prospect theory,
            using an external weighting function from the ProspectUtility class.
            """
            # Extract parameter values
            alpha = copy.deepcopy(parameters.alpha.value)
            lambd = copy.deepcopy(parameters.lambda_loss.value)
            beta = copy.deepcopy(parameters.beta.value)  # This is used for the utility curvature
            gamma = copy.deepcopy(parameters.gamma.value)  # This is used as the weighting curvature for gains
            delta = copy.deepcopy(parameters.delta.value)  # This is used for the weighting curvature for losses
            temperature  = copy.deepcopy(parameters.temperature.value)
            safe_magnitude = trial["safe_magnitudes"]
            risky_magnitude = trial["risky_magnitudes"]
            risky_prob = trial["risky_probability"]
            observed = trial["observed"].astype(int)
            # Compute objective expected values (EV)
            ev_safe = safe_magnitude
            ev_risk = risky_magnitude * risky_prob

            # Determine which option is objectively better
            objective_best = 1 if ev_risk >= ev_safe else 0

            # Create a temporary instance; dummy magnitudes are provided (they're not used in weighting)
            # Now use our unified parameter names: alpha for utility curvature, lambda_loss, and gamma
            pt_util = ProspectUtility(
                magnitudes=np.array([[safe_magnitude], [risky_magnitude]]),
                probabilities=np.array([[1.0], [risky_prob]]),
                alpha=alpha,
                beta=beta,
                lambda_loss=lambd,
                gamma=gamma,
                delta=delta,
                weighting=parameters.weighting,
                utility_curve=parameters.utility_curve,
            )

            utilities = pt_util.compute()

            # Compute softmax probabilities using the specified temperature
            sm = Softmax(temperature=temperature, activations=utilities)
            policies = sm.compute()

            # Determine choice: generate a response if required, else use the observed one
            chosen = sm.choice()

            # Determine if the chosen option is optimal
            is_optimal = 1 if chosen == objective_best else 0
            prob_chosen = policies[1]

            return {
                "policy": policies,
                "dependent": np.array([prob_chosen]),
                "observed": observed,  # Ensure the optimizer sees the 'observed' column
                "chosen": chosen,
                "is_optimal": is_optimal,
                "objective_best": objective_best,
                "ev_safe": ev_safe,
                "ev_risk": ev_risk,
                "u_safe": utilities[0],
                "u_risk": utilities[1],
            }

        # Pass the model function and parameters to the parent Wrapper
        super().__init__(data=data, model=model_fn, parameters=params)

class PTSM2025(Wrapper):
    """
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

    ### Model Description

    In what follows, we briefly describe the model's operations. First, the model calculates the subjective probability of the risky option, adjusting for ambiguity aversion using the parameter `eta`, denoted with $\\eta$. The subjective probability is computed as:

    $$
    p_{subjective} = p_{risky} - \\eta \\cdot ambiguity
    $$

    where $p_{risky}$ is the original probability of the risky choice and $ambiguity$ is the ambiguity associated with the risky option, either 0 for non-ambiguous or 1 for ambiguous cases.
    The utility of the safe and risky options is then computed using a utility function, which can be either a power function or a user-defined utility curve.
    If a utility curve is not provided, the model uses the following power function with curvature parameter `alpha`, denoted with $\\alpha$:

    $$
    u(x) =
    \\begin{cases}
        x^\\alpha & \\text{if } x \\geq 0 \\\\
        -|x|^\\alpha & \\text{if } x < 0
    \\end{cases}
    $$

    The model then applies loss aversion and gain sensitivity adjustments based on the sign of the risky choice magnitude. Here, the gain sensitivity `phi_gain`, denoted as $\\phi_{gain}$, is applied when the risky choice is positive, and the loss sensitivity `phi_loss`, denoted as $\\phi_{loss}$, is applied when the risky choice is negative. The adjusted probability of choosing the risky option, $p(A_{risky})$, is computed using a softmax function:

    $$
    p(A_{risky}) = \\frac{e^{\\beta (u_{risky} + \\phi_{t})}}{e^{\\beta (u_{risky} + \\phi_{t})} + e^{\\beta u_{safe}}}
    $$

    where denoted with $\\beta$ is the `temperature` parameter, $u_{risky}$ is the utility of the risky option, $u_{safe}$ is the utility of the safe option, and $\\phi_{t}$ is either $\\phi_{gain}$ or $\\phi_{loss}$ depending on the sign of the risky choice magnitude. Note that in Chew et al. (2019), the model only has a gambling bias term for the gain loss, that is then added to the difference between the safe and risky utilities, and only then transformed to a probability via a sigmoid function.

    Furthermore, the model generates a response based on the computed probabilities, where the choice is sampled from a Bernoulli distribution with the computed policy as the probability of choosing the risky option.

    ### Model Output

    For each trial, the model outputs the following variables:

        - `policy`: The computed probabilities for the risky options.
        - `model_choice`: The model's predicted choice (0 for safe, 1 for risky).
        - `real_choice`: The observed (participant's) choice from the data.
        - `u_safe`: The utility of the safe option.
        - `u_risk`: The utility of the risky option.
        - `dependent`: The computed probability of a risky choice according to the model, which can be used for further analysis or fitting
    

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
        if parameters_settings is None:
            warnings.warn("No parameters specified, using JAGS-inspired defaults.")
            parameters_settings = {
                "eta":         [0.0,   -0.49,  0.49],
                "phi_gain":    [0.0,   -10.0,  10.0],
                "phi_loss":    [0.0,   -10.0,  10.0],
                "temperature": [5.0,    0.001, 20.0],
                "alpha":       [1.0,    0.001,  5.0],
            }

        self.variant = variant

        if callable(utility_curve):
            warnings.warn("Utility curve provided, using it instead of power function.")
        if utility_curve is not None and not callable(utility_curve):
            raise ValueError("Utility curve must be a callable function.")

        def transform(x, alpha):
            ## Piecewise utility transform
            return x ** alpha if x >= 0 else -np.abs(x) ** alpha

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
            utility_curvature=transform,
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

        # CORRECTED: Renamed back to model_fn
        def model_fn(parameters, trial):
            eta = copy.deepcopy(parameters.eta)
            phi_gain = copy.deepcopy(parameters.phi_gain)
            phi_loss = copy.deepcopy(parameters.phi_loss)
            temperature = copy.deepcopy(parameters.temperature)
            alpha = copy.deepcopy(parameters.alpha)

            safe = trial["safe_magnitudes"]
            risky = trial["risky_magnitudes"]
            risky_probability= trial["risky_probability"]
            ambiguity  = trial["ambiguity"]
            observed = trial["observed"].astype(int)

            # Compute subjective probability with ambiguity aversion
            subjective_risky_probability = np.clip(risky_probability- eta * ambiguity, 0, 1)


            utility_safe_option  = parameters.utility_curvature(safe, alpha)
            utility_risky_option  = subjective_risky_probability * parameters.utility_curvature(risky, alpha)

            ## Adjust phi_t based on the sign of the magnitude of risky choice
            if risky >= 0:
                phi_t = phi_gain
            else:
                phi_t = phi_loss
            
            ## compute the policies adjusted via loss aversion and gain sensitivity
            policies = np.exp(temperature * utility_risky_option  + phi_t) / (
                np.exp(temperature * utility_risky_option  + phi_t) + np.exp(temperature * utility_safe_option)
            )
            ## generate a random response between 0 and 1
            model_choice = np.random.choice([0,1], p=[1-policies, policies])
            
            output = {
                "policy": policies,
                "model_choice": model_choice,
                "real_choice": observed,
                "u_safe": utility_safe_option,
                "u_risk": utility_risky_option,
                "dependent": np.array([policies])
            }
            
            return output

        super().__init__(data=data, model=model_fn, parameters=parameters)