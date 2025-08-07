import cpm
import numpy as np
import warnings
from scipy.stats import norm, truncnorm
from cpm.generators import Wrapper, Parameters, Value
from cpm.models.decision import Softmax

class PTSMExtended(Wrapper):
    """
    An extended Prospect Theory Softmax Model supporting:
      - Softmax bias (phi_gain / phi_loss)
      - Ambiguity aversion (eta)
      - Optional utility curvature (alpha)

    This class structure is designed for robust hierarchical fitting.
    """

    def __init__(
        self,
        data=None,
        parameters_settings=None,
        generate=False,
        mode="safe_risky",
        variant="1alpha"
    ):
        if parameters_settings is None:
            warnings.warn("No parameters specified, using JAGS-inspired defaults.")
            parameters_settings = {
                "alpha":     [1.0, 0.001, 5.0],
                "eta":       [0.0, -0.49, 0.49],
                "phi_gain":  [0.0, -10.0, 10.0],
                "phi_loss":  [0.0, -10.0, 10.0],
                "temperature": [5.0, 0.001, 20.0],
            }

        self.variant = variant
        self.mode = mode

        parameters = Parameters(
            eta=Value(
                value=parameters_settings["eta"][0],
                lower=parameters_settings["eta"][1],
                upper=parameters_settings["eta"][2],
                prior=norm,
                args={"loc": 0.0, "scale": 0.5}
            ),
            phi_gain=Value(
                value=parameters_settings["phi_gain"][0],
                lower=parameters_settings["phi_gain"][1],
                upper=parameters_settings["phi_gain"][2],
                prior=norm,
                args={"loc": 0.0, "scale": 0.5}
            ),
            phi_loss=Value(
                value=parameters_settings["phi_loss"][0],
                lower=parameters_settings["phi_loss"][1],
                upper=parameters_settings["phi_loss"][2],
                prior=norm,
                args={"loc": 0.0, "scale": 0.5}
            ),
            temperature=Value(
                value=parameters_settings["temperature"][0],
                lower=parameters_settings["temperature"][1],
                upper=parameters_settings["temperature"][2],
                prior=truncnorm,
                args={
                    "a": (parameters_settings["temperature"][1] - 5.0) / 2.5,
                    "b": (parameters_settings["temperature"][2] - 5.0) / 2.5,
                    "loc": 5.0, "scale": 2.5
                }
            ),
        )

        if variant == "1alpha":
            parameters["alpha"] = Value(
                value=parameters_settings["alpha"][0],
                lower=parameters_settings["alpha"][1],
                upper=parameters_settings["alpha"][2],
                prior=truncnorm,
                args={
                    "a": (parameters_settings["alpha"][1] - 1.0) / 1.0,
                    "b": (parameters_settings["alpha"][2] - 1.0) / 1.0,
                    "loc": 1.0, "scale": 1.0
                }
            )

        # CORRECTED: Renamed back to model_fn
        def model_fn(parameters, trial, generate=generate):
            # CORRECTED: Access parameters directly, without .value
            eta = parameters.eta
            phi_gain = parameters.phi_gain
            phi_loss = parameters.phi_loss
            temp = parameters.temperature
            alpha = parameters.alpha if variant == "1alpha" else 1.0

            safe = trial["safe_magnitudes"]
            risky = trial["risky_magnitudes"]
            prob = trial["risky_probability"]
            ambig = trial["ambiguity"]
            observed = int(trial["observed"])

            #subj_prob = prob - eta * ambig
            subj_prob = np.clip(prob - eta * ambig, 0, 1)

            def transform(x):
                return x ** alpha if x >= 0 else -np.abs(x) ** alpha

            u_safe = transform(safe)
            u_risk = subj_prob * transform(risky)

            phi_t = phi_gain if risky >= 0 else phi_loss
            u_risk += phi_t
            
            if self.mode == "safe_risky":
                activations = np.array([u_safe, u_risk])
            else:
                ev_safe = safe * 1.0
                ev_risk = risky * prob
                objective_best = 1 if ev_risk >= ev_safe else 0
                activations = np.array([u_risk, u_safe]) if objective_best == 0 else np.array([u_safe, u_risk])
            
            sm = cpm.models.decision.Softmax(
                activations=activations.reshape(2,1), temperature=temp
            )
            policies = sm.compute()
            
            # This is the probability of the choice the participant actually made.
            prob_chosen = policies[observed]
            
            # This simulates the model's own choice based on its policies.
            model_choice = np.random.choice(np.arange(len(policies)), p=policies.flatten())
            
            # CORRECTED: The full, well-formatted output dictionary
            output = {
                "policy": policies,
                "model_choice": model_choice,
                "real_choice": observed,
                "dependent": np.array([prob_chosen])
            }
            
            return output

        super().__init__(data=data, model=model_fn, parameters=parameters)