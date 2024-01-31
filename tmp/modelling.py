from cpm.models import Parameters, Wrapper, Simulator
from cpm.optimisation import DifferentialEvolution, minimise
from cpm.evaluation import strategies, ParameterRecovery

# import components we want to use
from cpm.components.utils import Nominal
from cpm.components.activation import LinearActivation
from cpm.components.learning import DeltaRule

# import some useful stuff
from prettyformatter import pprint as pp  ## pretty print
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt

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

# activations = np.array([[0.7, 0, -0.1], [-0.3, 0, 0.3]])


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
    ## multiply weights with stimulis vector
    active = LinearActivation(input=stimulus, weights=weights)
    active.compute()
    ## this is a mock-up policy
    policy = np.sum(active.weights) * temperature
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

params = {
    "alpha": 0.4,
    "temperature": 1,
    "values": np.zeros((1, 4)),
}

params = Parameters(**params)

print(params)
params.export()

wrap = Wrapper(model=model, parameters=params, data=data)

wrap.reset([0.5, 1])

wrap.run()
wrap.policies

wrap.reset(parameters={"alpha": 0.5, "temperature": 1, "values": np.zeros((1, 4))})
wrap.parameters

experiment = []
for i in range(100):
    ppt = {
        "ppt": i,
        "trials": np.array(
            [
                [2, 3],
                [1, 4],
                [3, 2],
                [4, 1],
                [2, 3],
                [2, 3],
                [1, 4],
                [3, 2],
                [4, 1],
                [2, 3],
            ]
        ),
        "feedback": np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1]),
        "observed": np.array([[1], [0], [1], [0], [1], [1], [0], [1], [0], [1]]),
    }
    experiment.append(ppt)


generator = Simulator(model=wrap, parameters=params.export(), data=experiment)
pp(generator.parameters)
generator.run()
pp(generator.simulation)

generator.policies()

policies = generator.policies()

policies

lower = [1e-10, 1e-10]
upper = [1, 1]
bounds = list(zip(lower, upper))

# FIXME: this is not working, metric is always overwritten
Fit = DifferentialEvolution(
    model=wrap,
    bounds=bounds,
    data=experiment,
    minimisation="LogLikelihood",
    mutation=0.5,
    recombination=0.7,
    strategy="best1bin",
    tol=0.1,
    maxiter=200,
)  # initialize the optimisation

Fit.minimise(np.array([0.1398096, 0.16551498]))
Fit.minimise([0.23360573, 0.81339223])

Fit.optimise()

pp(Fit.fit)
pp(Fit.parameters)


sims = Simulator(model=wrap, data=experiment, parameters=Fit.parameters)

sims.run()
sims.policies()

new = Simulator(model=wrap, data=experiment, parameters=Fit.parameters)

recover = ParameterRecovery(
    model=new,
    strategy="grid",
    optimiser="DifferentialEvolution",
    minimasation="LogLikelihood",
    bounds=bounds,
    iteration=1,
)

recover.recover()

import matplotlib.pyplot as plt

alpha = recover.extract(key="alpha")

recovered = np.asarray(alpha[:, 0, :]).flatten()
original = np.asarray(alpha[:, 1, :]).flatten()

plt.scatter(recovered, original)
plt.show()

pp(recover.output[0].get("original"))
