import sys
import os
import pandas as pd
import numpy as np
import pickle as pkl
from prettyformatter import pprint
from cpm.generators import Parameters, Wrapper, Value
from cpm.models import activation, decision

# get the directory of the current file, and add to sys.path if necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# load the curated dataset, *assuming it's in `current_dir`*
filename = os.path.join(current_dir, "scav_curated_data_debug.pkl")
curated_data = pkl.load(open("tmp/scav_curated_data_debug.pkl", "rb"))

print("\nExample of curated data:")
pprint(curated_data[0])

# set up a simple model with risk attitude parameter 'alpha' and choice
# temperature 'tau':

# Define the parameter set. All defined on the real line (-Inf, Inf), to be
# transformed to suitably bounded range within the model function.
params = {
    "alpha": 0.75,  # risk attitude
    "tau": -0.5,  # choice temperature
    "values": np.zeros((1, 2)),
}

params = Parameters(
    alpha=Value(
        value=-0.75,
        lower=-3,
        upper=3,
        prior="truncated_normal",
        args={"mean": 0, "sd": 1},
    ),
    tau=Value(
        value=-0.5,
        lower=-3,
        upper=3,
        prior="truncated_normal",
        args={"mean": 0, "sd": 1},
    ),
    values=np.zeros((1, 2)),
)


# Define the model function.
def model(parameters, trial):

    # transform alpha and tau parameter values from the real line (-Inf, Inf)
    # to parameter-specific lower- and upper-bounded ranges.
    def unbounded_to_bounded(x, lb, ub):
        # transform real value x to proportion, using inverse logit
        inv_logit = (1 / 2) + (1 / 2) * np.tanh(x / 2)
        # apply lower and upper bounds to proportion, and return
        return lb + (ub - lb) * inv_logit

    alpha = unbounded_to_bounded(x=parameters["alpha"], lb=0.1, ub=3)
    tau = unbounded_to_bounded(x=parameters["tau"], lb=0.001, ub=2)

    # print(alpha)
    # print(tau)

    # compute the expected utility of each choice option
    utilities = activation.ProspectUtility(
        magnitudes=trial.get("magnitudes"),
        probabilities=trial.get("probabilities"),
        alpha_pos=alpha,
    )
    utilities.compute()
    expected_utilities = utilities.expected_utility

    # reshape the utilities array to work with the Softmax class
    final_utilities = expected_utilities.reshape(2, 1)
    # subtract the maximum from the utilities to improve numerical stability
    final_utilities_shifted = final_utilities - np.max(final_utilities)

    # transform the expected utilities into choice probabilities with the
    # softmax function
    choiceprobs = decision.Softmax(
        activations=final_utilities_shifted, temperature=1 / tau
    )
    choiceprobs.compute()

    # simulate a choice based on the probabilities
    choice = choiceprobs.choice()

    # determine the index of the risky option in the current trial
    risky_index = np.where(trial.get("trials") == 2)[0]

    # compile output and return
    output = {
        "utility": final_utilities,
        "policy": choiceprobs.policies,
        "choice": choice,
        "values": choiceprobs.policies,
        # `dependent` is the key variable for model fitting, to be compared
        # to an `observed`` field in the data dictionary
        # since the observed choice is coded as safe = 0 and risky = 1, the
        # only value we need is the probability of choosing the risky option
        "dependent": np.array([choiceprobs.policies[risky_index]]),
    }

    return output


# define model wrapper function, arbitrarily using the first entry of the data
# list as a template
wrap = Wrapper(model=model, parameters=params, data=curated_data[0])
wrap.run()

# if __name__ == "__main__":

from cpm.optimisation import minimise, Bads

np.seterr(all="ignore")
# define the model fitting function
Fit = Bads(
    model=wrap,
    data=curated_data[0:2],
    minimisation=minimise.LogLikelihood.bernoulli,
    initial_guess=None,
    number_of_starts=2,
    parallel=True,
    options={"display": "off"},
)

# run the optimisation
Fit.optimise()

test = EmpiricalBayes(optimiser=Fit, iteration=2, tolerance=1e-6, chain=2)
