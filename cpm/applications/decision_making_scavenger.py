import numpy as np
import warnings
from cpm.generators import Wrapper, Parameters, Value
from cpm.models.decision import Softmax

class PTSMExtended(Wrapper):
    """
    An extended Prospect Theory Softmax Model supporting:
      - Softmax bias (phi_gain / phi_loss)
      - Ambiguity aversion (eta)
      - Optional utility curvature (alpha)

    Supports two variants:
        - '1alpha': includes alpha
        - 'noalpha': fixes alpha = 1 (i.e., linear utility)
    """

    def __init__(
        self,
        data=None,
        parameters_settings=None,
        generate=False,
        mode="safe_risky",
        variant="1alpha"  # or "noalpha"
    ):
        if parameters_settings is None:
            warnings.warn("No parameters specified, using JAGS-inspired defaults.")
            parameters_settings = {
                # format: [start_value, lower_bound, upper_bound]
                "alpha":     [1.0, 0.1, 2.0],
                "eta":       [0.0, -2.0, 2.0],
                "phi_gain":  [0.0, -5.0, 5.0],
                "phi_loss":  [0.0, -5.0, 5.0],
                "temperature": [5.0, 0.001, 20.0],
            }

        self.variant = variant
        self.mode = mode

        parameters = Parameters(
            # JAGS: eta_mu ~ dunif(-2, 2)
            eta=Value(
                value=parameters_settings["eta"][0],
                lower=-2.0, upper=2.0,
                prior="normal",  # Can be negative, so not truncated
                args={"mean": 0.0, "sd": 1.0}
            ),
            
            # JAGS: phi_gain_mu ~ dunif(-10, 10)
            phi_gain=Value(
                value=parameters_settings["phi_gain"][0],
                lower=-5.0, upper=5.0,
                prior="normal",
                args={"mean": 0.0, "sd": 1.0}
            ),
            
            # JAGS: phi_loss_mu ~ dunif(-10, 10)
            phi_loss=Value(
                value=parameters_settings["phi_loss"][0],
                lower=-5.0, upper=5.0,
                prior="normal",
                args={"mean": 0.0, "sd": 1.0}
            ),
            
            # JAGS: softmax_beta_mu ~ dunif(0, 10), and T(0.001,)
            temperature=Value(
                value=parameters_settings["temperature"][0],
                lower=0.001, upper=20.0,
                prior="truncated_normal",
                args={"mean": 5.0, "sd": 2.5}
            ),
        )

        if variant == "1alpha":
            # JAGS: alpha_mu ~ dunif(0, 5), and T(0.001,)
            parameters["alpha"] = Value(
                value=parameters_settings["alpha"][0],
                lower=0.001, upper=5.0,
                prior="truncated_normal",
                args={"mean": 1.0, "sd": 1.0}
            )

        def model_fn(parameters, trial, generate=generate):
            eta = parameters.eta.value
            phi_gain = parameters.phi_gain.value
            phi_loss = parameters.phi_loss.value
            temp = parameters.temperature.value
            alpha = parameters.alpha.value if variant == "1alpha" else 1.0

            safe = trial["safe_magnitudes"]
            risky = trial["risky_magnitudes"]
            prob = trial["risky_probability"]
            ambig = trial["ambiguity"]
            obs = int(trial["observed"])

            ev_safe = safe
            ev_risk = risky * prob
            objective_best = 1 if ev_risk >= ev_safe else 0

            # Apply ambiguity aversion to probability
            subj_prob = prob - eta * ambig

            def transform(x):
                return x ** alpha if x >= 0 else -abs(x) ** alpha

            u_safe = transform(safe)
            u_risk = subj_prob * transform(risky)

            # Apply domain-specific logit bias
            phi_t = phi_gain if risky >= 0 else phi_loss
            u_risk += phi_t

            # Decide softmax input order
            if self.mode == "safe_risky":
                activations = np.array([u_safe, u_risk])
            elif self.mode == "optimal_nonoptimal":
                activations = np.array([u_risk, u_safe]) if objective_best == 0 else np.array([u_safe, u_risk])
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            sm = Softmax(temperature=temp, activations=activations.reshape(2, 1))
            policies = sm.compute()
            chosen = sm.choice() if generate else obs
            is_optimal = 1 if chosen == objective_best else 0
            prob_chosen = policies[chosen]

            return {
                "policy": policies,
                "dependent": np.array([prob_chosen]),
                "observed": obs,
                "chosen": chosen,
                "is_optimal": is_optimal,
                "objective_best": objective_best,
                "ev_safe": ev_safe,
                "ev_risk": ev_risk,
                "u_safe": u_safe,
                "u_risk": u_risk,
            }

        super().__init__(data=data, model=model_fn, parameters=parameters)