import copy
import numpy as np
import warnings
from cpm.generators import Wrapper, Parameters, Value
from cpm.models.decision import Softmax
from cpm.models.activation import ProspectUtility

class PTSM(Wrapper):
    """
    A Prospect Theory-based Softmax Model (PTSM) for decision-making tasks based on Kahneman & Tversky (1979). It computes expected utility by combining transformed magnitudes and weighted probabilities, suitable for safe–risky decision paradigms.

    The model can interpret data either as:
        - 'safe vs. risky' (choice=0 => safe, choice=1 => risky)
        OR
        - 'optimal vs. non-optimal' (choice=0 => suboptimal, 1 => optimal)  #this is only here in preperation to a specific form of joint modeling 
    
    The model computes objective EV internally (ev_safe vs. ev_risk)
    and outputs trial-level information (including whether the chosen option is optimal).
    
    Additionally, the model accepts a "weighting" argument that determines 
    which probability weighting function to use when computing the subjective 
    weighting of risky probabilities.
    """

    def __init__(
        self,
        data=None,
        parameters_settings=None,
        generate=False,
        mode="safe_risky",
        weighting="tk"  # Options: "tk" or "power"
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            The data, where each row is a trial and each column is an input to the model. Expected to have columns: 'safe_magnitudes', 'risky_magnitudes', 'risky_probability', 'observed'
        parameters_settings : list of lists
            e.g. [
                [alpha0, alpha_low, alpha_high],
                [lambda0, lam_low, lam_high],
                [gamma0, g_low,   g_high],
                [temp0,  t_low,   t_high],
                ]
        generate : bool
            If True, the model will sample from the softmax distribution instead of using the observed 'choice'.
        mode : str
            "safe_risky" (default) interprets option 0 as safe and option 1 as risky.
            "optimal_nonoptimal" interprets choices based on objective EV.
        weighting : str
            The probability weighting function to use. Options include:
                "power": use a simple power function (p^gamma)
                "tk": use the Tversky–Kahneman (1992) weighting function.
                See [cpm.models.activation.ProspectUtility](cpm.models.activation.ProspectUtility) .

        See Also
        ---------
        [cpm.models.decision.Softmax](cpm.models.decision.Softmax) : for mapping utilities to choice probabilities.

        [cpm.models.activation.ProspectUtility](cpm.models.activation.ProspectUtility) : for the Prospect Utility class that computes subjective utilities and weighted probabilities.

        References
        ----------

        Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2), 263–291.

        """
        # Use default parameter settings if none provided.
        if parameters_settings is None:
            parameters_settings = [
                [1.0, 0.0, 5.0],   # alpha: starting value 1.0
                [1.0, 0.0, 5.0],   # lambda_loss: starting value 1.0
                [0.5, 0.0, 5.0],   # gamma: starting value 0.5 (changed to not equal 1.0)
                [5.0, 0.1, 15.0]   # temperature: starting value 5.0
            ]
            warnings.warn("No parameters specified, using default settings.")

        # Create the unified Parameters object with priors.
        params = Parameters(
            alpha=Value(
                value=parameters_settings[0][0],
                lower=parameters_settings[0][1],
                upper=parameters_settings[0][2],
                prior="truncated_normal",
                args={"mean": parameters_settings[0][0], "sd": 0.25},
            ),
            lambda_loss=Value(
                value=parameters_settings[1][0],
                lower=parameters_settings[1][1],
                upper=parameters_settings[1][2],
                prior="truncated_normal",
                args={"mean": parameters_settings[1][0], "sd": 0.25},
            ),
            gamma=Value(
                value=parameters_settings[2][0],
                lower=parameters_settings[2][1],
                upper=parameters_settings[2][2],
                prior="truncated_normal",
                args={"mean": parameters_settings[2][0], "sd": 0.25},
            ),
            temperature=Value(
                value=parameters_settings[3][0],
                lower=parameters_settings[3][1],
                upper=parameters_settings[3][2],
                prior="truncated_normal",
                args={"mean": parameters_settings[3][0], "sd": 2.5},
            ),
        )

        self.mode = mode
        self.weighting = weighting  # Store the chosen weighting function type

        def model_fn(parameters, trial, generate=generate):
            """
            Called per trial. Computes the subjective utility for two options based on prospect theory,
            using an external weighting function from the ProspectUtility class.
            """
            # Extract parameter values
            alpha = parameters.alpha.value
            lambd = parameters.lambda_loss.value
            gamma = parameters.gamma.value  # This is used as the weighting curvature
            temp  = parameters.temperature.value

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

            # Use the existing ProspectUtility weighting function:
            # Import the ProspectUtility class from activations_ben
            # Create a temporary instance; dummy magnitudes are provided (they're not used in weighting)
            dummy_magnitudes = np.array([1])
            dummy_probabilities = np.array([risky_prob])
            # Now use our unified parameter names: alpha for utility curvature, lambda_loss, and gamma
            pt_util = ProspectUtility(
                magnitudes=dummy_magnitudes,
                probabilities=dummy_probabilities,
                alpha=alpha,
                lambda_loss=lambd,
                gamma=gamma,
                weighting=self.weighting
            )
            # Compute the weighted probability using the public method
            p_weighted = pt_util.weight_probability(risky_prob)

            # Define a simple piecewise utility transform
            def transform(x):
                if x >= 0:
                    return x ** alpha
                else:
                    return -lambd * (abs(x) ** alpha)

            u_safe = transform(safe_magn)
            u_risk = p_weighted * transform(risky_magn)

            # Determine activations based on mode
            if self.mode == "safe_risky":
                activations = np.array([u_safe, u_risk])
            elif self.mode == "optimal_nonoptimal":
                activations = np.array([u_risk, u_safe]) if objective_best == 0 else np.array([u_safe, u_risk])
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            # Compute softmax probabilities using the specified temperature
            sm = Softmax(temperature=temp, activations=activations.reshape(2, 1))
            policies = sm.compute()

            # Determine choice: generate a response if required, else use the observed one
            chosen = sm.choice() if generate else observed

            # Determine if the chosen option is optimal
            is_optimal = 1 if chosen == objective_best else 0
            prob_chosen = policies[chosen]

            return {
                "policy": policies,
                "dependent": np.array([prob_chosen]),
                "observed": observed,  # Ensure the optimizer sees the 'observed' column
                "chosen": chosen,
                "is_optimal": is_optimal,
                "objective_best": objective_best,
                "ev_safe": ev_safe,
                "ev_risk": ev_risk,
                "u_safe": u_safe,
                "u_risk": u_risk,
            }

        # Pass the model function and parameters to the parent Wrapper
        super().__init__(data=data, model=model_fn, parameters=params)

class PTSMExtended(Wrapper):
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
        An instance of the PTSMExtended model, which can be used to fit data and generate predictions.

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

    The model then applies loss aversion and gain sensitivity adjustments based on the sign of the risky choice magnitude. Here, the gain sensitivity `phi_gain`, denoted as $\\phi_{gain}$, is applied when the risky choice is positive, and the loss sensitivity `phi_loss`, denoted as $\\phi_{loss}$, is applied when the risky choice is negative. The adjusted policies are computed using a softmax function:

    $$
    p(A_{risky}) = \\frac{e^{\\beta (u_{risky} + \\phi_{t})}}{e^{\\beta (u_{risky} + \\phi_{t})} + e^{\\beta u_{safe}}}
    $$

    where denoted with $\\beta$ is the `temperature` parameter, $u_{risky}$ is the utility of the risky option, $u_{safe}$ is the utility of the safe option, and $\\phi_{t}$ is either $\\phi_{gain}$ or $\\phi_{loss}$ depending on the sign of the risky choice magnitude. Note that in Chew et al. (2019), the model only has a gambling bias term for the gain loss, that is then added to the difference between the safe and risky utilities, and only then transformed to a probability via a sigmoid function.

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
            model_choice = np.random.choice([0,1], p=[policies, 1-policies])
            
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