from cpm.generators import Parameters, Wrapper, Value

# import components we want to use
from cpm.models.utils import Nominal
from cpm.models.learning import DeltaRule
from cpm.models.decision import Sigmoid

# import some useful stuff
import numpy as np


## create a set of parameters
## the parameter class is only used for the Wrapper
## other classes and modules will convert the parameters to this class automatically
parameters = Parameters(
    # freely varying parameters are indicated by specifying priors
    alpha=Value(
        value=0.5,
        lower=1e-10,
        upper=1,
        prior="truncated_normal",
        args={"mean": 0.5, "sd": 0.25},
    ),
    temperature=Value(
        value=1,
        lower=1e-10,
        upper=10,
        prior="truncated_normal",
        args={"mean": 5, "sd": 2.5},
    ),
    values=np.array([[0.25, 0.25, 0.25, 0.25]]),
)


## create a single trial as a dictionary
trial = {
    "trials": np.array([1, 2]),
    "feedback": np.array([1]),
    "attention": np.array([1, 0]),  ## not used in the model
    "misc": np.array([1, 0]),  ## not used in the model
}


## define a quick model
def model(parameters, trial):
    ## import parameters
    alpha = parameters.alpha
    temperature = parameters.temperature
    ## import variables
    weights = parameters.values

    ## import trial
    stimulus = trial.get("trials")
    stimulus = Nominal(target=stimulus, bits=4)
    feedback = trial.get("feedback")

    ## specify the computations for a given trial
    ## multiply weights with stimulis vector
    active = stimulus * weights
    ## this is a mock-up policy
    policy = Sigmoid(temperature=temperature, beta=0.5, activations=active).compute()
    ## learning
    error = DeltaRule(
        weights=active, feedback=feedback, alpha=alpha, input=stimulus
    ).compute()
    weights += error
    output = {
        "policy": np.array([policy]),
        "values": weights,
        "dependent": np.array([policy]),
    }
    return output


def test():
    model(parameters, trial)


import timeit

print(timeit.timeit("test()", setup="from __main__ import test", number=10000))
print(timeit.timeit("test()", globals=locals(), number=1))

## create some data
## could be in a separate script
experiment = []
for i in range(200):
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
        "feedback": np.random.randint(2, size=(10, 1)),
        "observed": np.random.randint(2, size=(10, 1)),
    }
    experiment.append(ppt)


## reset wrapper for no reason other than tidying
## Wrapper can only take a single dictionary as data, so experiment[0] is used
wrap = Wrapper(model=model, parameters=parameters, data=experiment[0])


from cpm.optimisation import minimise, Fmin

## initialise the optimisation object
Fit = Fmin(
    model=wrap,
    data=experiment,
    initial_guess=[0.32, 0.788],
    minimisation=minimise.LogLikelihood.bernoulli,
    parallel=False,  # parallel True will use all available cores
)

# run the optimisation
Fit.optimise()

## export data where each row is a participant
Fit.export()

Fit.parameters
