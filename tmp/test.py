from cpm.generators import Value, Parameters, Wrapper

# import components we want to use

# import components we want to use
from cpm.models.utils import Nominal
from cpm.models.learning import DeltaRule

# import some useful stuff
from prettyformatter import pprint as pp  ## pretty print
import numpy as np


## create a single trial as a dictionary
trial = {
    "trials": np.array([1, 2]),
    "feedback": np.array([1]),
    "attention": np.array([1, 0]),  ## not used in the model
    "misc": np.array([1, 0]),  ## not used in the model
}

## create a set of parameters
## the parameter class is only used for the Wrapper
## other classes and modules will convert the parameters to this class automatically
parameters = Parameters(
    alpha=0.1, temperature=1, values=np.array([[-0.5, 0.7, 0.9, 0]])
)


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
    policy = np.sum(active) * 1 / temperature
    ## learning
    learning = DeltaRule(weights=active, feedback=feedback, alpha=alpha, input=stimulus)
    learning.compute()
    error = learning.weights
    weights += error
    output = {
        "policy": np.array([policy]),
        "values": weights,
        "dependent": np.array([policy]),
    }
    return output


model(parameters, trial)


data = {
    "trials": np.array(
        [[2, 3], [1, 4], [3, 2], [4, 1], [2, 3], [2, 3], [1, 4], [3, 2], [4, 1], [2, 3]]
    ),
    "feedback": np.array([[1], [0], [1], [0], [1], [1], [0], [1], [0], [1]]),
}


params = Parameters(
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
        upper=1,
        prior="truncated_normal",
        args={"mean": 0.5, "sd": 0.25},
    ),
    values=np.array([[0.25, 0.25, 0.25, 0.25]]),
)

wrap = Wrapper(model=model, parameters=params, data=data)
wrap.run()
wrap.dependent


wrap.export()


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
        "feedback": np.array([[1], [0], [1], [0], [1], [1], [0], [1], [0], [1]]),
        "observed": np.random.randint(0, 2, 10),
    }
    experiment.append(ppt)


from cpm.optimisation import DifferentialEvolution, minimise, FminBound, Fmin


Fit = Fmin(
    model=wrap,
    data=experiment,
    initial_guess=[0.32, 0.5],
    minimisation=minimise.LogLikelihood.continuous,  # currently, this is the only working metric
    parallel=True,
    prior=False,
)

Fit.optimise()

pp(Fit.parameters)

Fit.export()

Fit.export()

test = EmpiricalBayes(optimiser=Fit, iteration=5, tolerance=1e-6, chain=2)

test.optimise()

x = test.optimiser.model.parameters.alpha.prior.kwds

test.details[0]

test.lmes
