import pandas as pd
import numpy as np
import warnings
import time
import cpm
import ipyparallel as ipp

from cpm.generators import Wrapper, Parameters, Value


class PUTM(Wrapper):
    """
    The Prospect Utility Theory model (PUTM) is a model of decision making under risk that is based on Kahneman and Tversky (1992).

    Parameters
    ----------
    data: pandas.DataFrame
        The data to be fit by the model. The data must contain columns for the choice and reward for each dimension. See Notes for more information on what columns should you include.
    dimensions: int
        The number of distinct stimuli present in the data.
    parameters_settings: list-like
        The parameters to be fit by the model. The parameters must be specified as a list of lists, with each list containing the value, lower, and upper bounds of the parameter. See Notes for more information on how to specify parameters and for the default settings.

    Notes
    -----

    """

    def __init__(self, data=None, parameters_settings=None, generate=False):

        if parameters_settings is None:
            parameters_settings = [
                [0.5, 0, 2],
                [5, -50, 50],
                [10, -100, 100],
                [1, 0, 20],
                [0, 0, 1],
            ]
            warnings.warn("No parameters specified, using default parameters.")

        #  "alpha" : risk attitude
        # "kappa"  : gambling bias
        # "gamma"  : response repetition bias
        # "tau"    : choice temperature
        # "xi"     : irreducible noise

        parameters = {
            "alpha": Value(
                value=parameters_settings[0][0],
                lower=parameters_settings[0][1],
                upper=parameters_settings[0][2],
                prior="truncated_normal",
                args={"mean": 0.5, "sd": 0.25},
            ),
            "kappa": Value(
                value=parameters_settings[1][0],
                lower=parameters_settings[1][1],
                upper=parameters_settings[1][2],
                prior="truncated_normal",
                args={"mean": 5, "sd": 10},
            ),
            "gamma": Value(
                value=parameters_settings[2][0],
                lower=parameters_settings[2][1],
                upper=parameters_settings[2][2],
                prior="truncated_normal",
                args={"mean": 0, "sd": 20},
            ),
            "tau": Value(
                value=parameters_settings[3][0],
                lower=parameters_settings[3][1],
                upper=parameters_settings[3][2],
                prior="truncated_normal",
                args={"mean": 1, "sd": 2.5},
            ),
            "xi": Value(
                value=parameters_settings[4][0],
                lower=parameters_settings[4][1],
                upper=parameters_settings[4][2],
                prior="truncated_normal",
                args={"mean": 0, "sd": 0.5},
            ),
        }

        @ipp.require("numpy")
        def model(parameters, trial):

            # define the expected utility of each choice option
            utilities = cpm.models.activation.ProspectUtility(
                magnitudes=trial.get("magnitudes"),
                probabilities=trial.get("probabilities"),
                alpha_pos=parameters.alpha,
            )
            utilities.compute()

            # determine the indices of the safe option and risky option
            safe_index = np.where(trial.get("trials") == 1)[0]
            risky_index = np.where(trial.get("trials") == 2)[0]

            # modulate the expected utilities with the gambling bias
            # specifically, a value 'kappa' is added to the expected utility of the
            # risky option.
            gamble = cpm.models.activation.Offset(
                input=utilities.expected_utility,
                offset=parameters.kappa,
                index=risky_index,
            )
            gamble.compute()

            # additionally modulate the utilities with the response reptition bias
            # specifically, a value 'gamma' is added to the expected utility of the
            # response option that is consistent with the previous trial's response.
            # for example, if on the previous trial the left response button was
            # pressed, then for the current trial the expected utility of whichever
            # option is presented on the left will be increased by 'gamma'.
            if trial.get("resp_prev") == -1:
                # for the first trial, there is no previous response (resp_prev == -1),
                # so the offset value is set to zero and we use an arbitrary option
                # index
                resp_repbias_offset = 0
                resp_repbias_index = safe_index
            else:
                # otherwise, the offset value is set to the parameter gamma, and we add
                # it to the option (risky or safe) that is consistent with the previous
                # response.
                resp_repbias_offset = parameters.gamma
                if (trial.get("resp_prev") == 0 and safe_index == 0) or (
                    trial.get("resp_prev") == 1 and safe_index == 1
                ):
                    resp_repbias_index = safe_index
                else:
                    resp_repbias_index = risky_index

            utilities_resp_repbias = cpm.models.activation.Offset(
                input=gamble.output,
                offset=resp_repbias_offset,
                index=resp_repbias_index,
            )
            utilities_resp_repbias.compute()

            # reshape the utilities array to work with the Softmax class
            final_utilities = utilities_resp_repbias.output.reshape(2, 1)

            # subtract the maximum from the utilities to improve numerical stability
            final_utilities_shifted = final_utilities - np.max(final_utilities)

            # calculate the policy (choice probabilities) based on choice utilities
            choiceprobs = cpm.models.decision.Softmax(
                activations=final_utilities_shifted,
                temperature=1 / parameters.tau,
                xi=parameters.xi,
            )
            choiceprobs.irreducible_noise()

            # simulate a choice based on the policy
            choice = choiceprobs.choice()

            # compile output
            output = {
                # define key model outputs
                "utility": final_utilities,
                "policy": choiceprobs.policies,
                "choice": choice,
                # duplicating policies array to ensure "values" field is included
                "values": choiceprobs.policies,
                # define the key variable for model fitting, to be compared against
                # an "observed" field in the data dictionary
                # since the observed choice is coded as safe = 0 and risky = 1, the
                # only value we need is the probability of choosing the risky option
                "dependent": np.array([choiceprobs.policies[risky_index]]),
            }

            return output

        super().__init__(data=data, model=model, parameters=parameters)
