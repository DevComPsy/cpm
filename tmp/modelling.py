from cpm.models import Parameters, Wrapper

# import components we want to use
from cpm.components.utils import Nominal
from cpm.components.activation import LinearActivation
from cpm.components.decision import Softmax, Sigmoid
from cpm.components.learning import DeltaRule

# import some useful stuff
from prettyformatter import pprint as pp  ## pretty print
import numpy as np
import pandas as pd


# data = pd.read_csv("~/Downloads/participant1.csv")

# data["ppt"] = 1

# data.rename(columns={"Feedback": "feedback", "Response": "response"}, inplace=True)

# result = PandasToDict(data, participant="ppt", stimuli="Stim", response="response")

# pp(result)

parameters = Parameters(alpha=0.1, temperature=1, values=np.array([-0.5, 0.7, 0.9]))
pp(parameters)

# trial = {
#     "input": np.array([1, 2]),
#     "feedback": np.array([1]),
#     "attention": np.array([1, 0]),
#     "misc": np.array([1, 0]),
# }

activations = np.array([[0.7, 0, -0.1], [-0.3, 0, 0.3]])

def model(parameters, trial):
    ## import parameters
    alpha = parameters.alpha
    temperature = parameters.temperature
    ## import variables
    weights = parameters.values

    ## import trial
    stimulus = trial.get("input")
    stimulus = Nominal(target=stimulus, bits=4)
    feedback = trial.get("feedback")

    ## specify the computations for a given trial
    active = LinearActivation(input=stimulus, weights=weights)
    active.compute()
    ## because you only have one outcome, you don't need softmax
    ## but only IF YOU USE THE DELTA LEARNING RULE WITH THE OUTCOME IN THE RANGE OF 0 AND 1.
    policy = np.sum(active.weights)
    ## learning
    learning = DeltaRule(
        weights=active.weights, feedback=feedback, alpha=alpha, input=stimulus
    )
    learning.compute()
    error = learning.weights
    weights += error
    output = {
        "policy": policy,
        "values": weights,
    }

    return output

# model(parameters, trial)

data = {
    "trials": np.array(
        [[2, 3], [1, 4], [3, 2], [4, 1], [2, 3], [2, 3], [1, 4], [3, 2], [4, 1], [2, 3]]
    ),
    "feedback": np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1]),
}

parameters = {
    "alpha": 0.4,
    "temperature": 5,
    "values": np.zeros((1, 4)),
    }

parameters = Parameters(**parameters)

print(parameters)
parameters.export()

x = Wrapper(model=model, parameters=parameters, data=data)

x.values
x.policies
x.model
x.feedback
x.training

x.run()
x.policies