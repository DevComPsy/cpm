# %%
from cpm.generators import Parameters, Wrapper

# import components we want to use

# import components we want to use
from cpm.models.utils import Nominal
from cpm.models.learning import DeltaRule

# import some useful stuff
from prettyformatter import pprint as pp  ## pretty print
import numpy as np


# %%
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


# %%
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
    policy = np.sum(active) * temperature
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


# %%
model(parameters, trial)


# %%
data = {
    "trials": np.array(
        [[2, 3], [1, 4], [3, 2], [4, 1], [2, 3], [2, 3], [1, 4], [3, 2], [4, 1], [2, 3]]
    ),
    "feedback": np.array([[1], [0], [1], [0], [1], [1], [0], [1], [0], [1]]),
}

# %%
# Add new parameters to the model
params = {
    "alpha": 0.5,
    "temperature": 1,
    "values": np.zeros((1, 4)),
}

params = Parameters(**params)

wrap = Wrapper(model=model, parameters=params, data=data)
wrap.run()
wrap.dependent

# %% [markdown]
# Awesome.
# We can also use the export function to export the model output with the parameters to a dictionary.
# The main difference between `export()` and `summary()` is that in the long run `export()` will contain more metadata and will allow to export to a JSON file.

# %%
pp(wrap.export())


# %%
wrap.reset(parameters={"alpha": 0.05, "temperature": 1, "values": np.zeros((1, 4))})
wrap.run()
pp(wrap.export())


# %%
wrap.simulation

# %%
wrap.save(filename="test")

# %%
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
        "observed": np.array([[1], [0], [1], [0], [1], [1], [0], [1], [0], [1]]),
    }
    experiment.append(ppt)


# %%
from cpm.optimisation import DifferentialEvolution, minimise

lower = [1e-10, 1e-10]
upper = [1, 1]
bounds = list(zip(lower, upper))

Fit = DifferentialEvolution(
    model=wrap,
    bounds=bounds,
    data=experiment,
    minimisation=minimise.LogLikelihood.continuous,  # currently, this is the only working metric
    mutation=0.5,  # kwargs
    recombination=0.7,  # kwargs
    strategy="best1bin",  # kwargs
    tol=0.1,  # kwargs
    maxiter=200,  # kwargs
)  # initialize the optimisation

Fit.optimise()

pp(Fit.fit)

# %%
pp(Fit.parameters)

# %%
Fit.export()

# %%
Fit.export(details=True)
# %% [markdown]
# This is great, so noew we can run the model on the best fitting parameters and see how it looks like.
# We can use the `Simulator` class to do that, which takes in the data as defined above a list of dictionaries for the parameter.
# I know that the Simulator was used a bit differently in what we outlined in the Engineering Document, but I think that this is still okay for now.

# %%
from cpm.generators import Simulator

wrap.reset()

explore = Simulator(model=wrap, data=experiment, parameters=Fit.parameters)
explore.run()


# %%
explore.simulation

# %%
explore.export()

# %% [markdown]
# Let's look at the output.
# We can simply export the outcome as a pandas dataframe and plot it.
# It is confusingly names `export()`, so that will need to be changed.

# %% [markdown]
# ### Evaluation
#
# Alright, let us now try to do some parameter recovery.
# There is less and less thing to explain now, so I will just go through it quickly.
#
# We will use the `ParameterRecovery` from `cpm.evaluation`.
# This will take in the Wrapper, the optimiser, the strategy, bounds, and iterations.

# %% [markdown]
# Okay, that was fairly quick.
# Let's look at the results for the learning rate, which we can simply extract from the `ParameterRecovery` object.
# NOTE that the model will have bad performance, but let's ignore that for now and focus on functionalities.

# %%
explore.generate()
explore.generated[0].get("observed").shape

# %%
from cpm.evaluation import ParameterRecovery, strategies


new = Simulator(model=wrap, data=experiment, parameters=Fit.parameters)

recover = ParameterRecovery(
    model=new,
    strategy=strategies.grid,
    optimiser=DifferentialEvolution,
    loss=minimise.LogLikelihood.continuous,
    iteration=10,
    bounds=bounds,
)

recover.recover()


# %%
recover.extract(key="alpha")

# %%
import matplotlib.pyplot as plt

alpha = recover.extract(key="alpha")
alpha["ppt"] = np.repeat(np.arange(100), 10)


# %%
alpha.plot.scatter(x="recovered", y="original", c="iteration", colormap="viridis", s=10)

# %% [markdown]
# ## The END
#
# That's it for now.
# There are some things that don't work as expected, but I am actively looking into them.
# I think the biggest challenge was the modular aspect of the toolbox, but I think that it is working quite well now.
