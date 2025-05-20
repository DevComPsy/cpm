import numpy as np
import warnings
from cpm.generators import Wrapper, Parameters, Value
from cpm.models.decision import Softmax
from cpm.models.activation import ProspectUtility

class PTSM(Wrapper):
    """
    A Prospect Theory-based Softmax Model (PTSM) for decision-making tasks based on (ref needed).
    The model can interpret data either as:
        - 'safe vs. risky' (choice=0 => safe, choice=1 => risky)
        OR
        - 'optimal vs. non-optimal' (choice=0 => suboptimal, 1 => optimal)
    
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

            # Read trial data (expected columns: safe_magn, risky_magn, risky_prob, choice)
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