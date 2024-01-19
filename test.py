from cpm.models import Model, Wrapper, Simulator
from cpm.models.zoo.activation import LinearActivation
from cpm.models.zoo.decision import Softmax
from cpm.models.zoo.learning import DeltaRule
from cpm.optimisation import DifferentialEvolution, minimise
from cpm.evaluation import strategies, ParameterRecovery
from prettyformatter import pprint as pp
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

data = {
        "trials": np.array([[2, 3], [1, 4], [3, 2], [4, 1], [2, 3], [2, 3], [1, 4], [3, 2], [4, 1], [2, 3]]),
        "feedback": np.array([1, 2, 1, 2, 1, 1, 2, 1, 2, 1]),
        "observed": np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]),
}

pp(data)

params = {  # define the parameters
    "alpha": 0.25,
    "temperature": 5,
}

arguments = {
    'alpha' : 0.1,
    'temperature': 1,
    'weights' : np.array([[0.5, 0.1], [0, 0.5]]),
    'input' : np.array([0, 1]),
    'feedback' : np.array([0, 1]),
    'attention' : np.array([1, 0]),
    'misc' : np.array([1, 0])
    }

input = arguments.copy()

# The code is creating an instance of the `Model` class and initializing it with a
# list of functions `[LinearActivation, Softmax, DeltaRule]`. Then, it calls the
# `compute` method of the `Model` instance, passing the `input` dictionary as
# keyword arguments. This method computes the model based on the provided
# arguments. Finally, it calls the `summary` method of the `Model` instance, which
# prints a summary of the model.

mine = Model(functions = [LinearActivation, Softmax, DeltaRule])
mine.compute(**input)
pp(mine.policy)
pp(mine.summary())
mine.values

pkl.dump(mine, open("mine.pkl", "wb"))
pickled_model = pkl.load(open("mine.pkl", "rb"))

pickled_model.policy
pp(pickled_model.values)
pickled_model.summary()

del pickled_model

x = Wrapper(model = mine, data = data, params = params)
x.run()
pp(x.summary())
pp(x.export())

x.reset()
pp(x.summary())
## set up multiple people
experiment = []
for i in range(100):
    ppt = {
        "ppt": i,
        "trials": np.array([[2, 3], [1, 4], [3, 2], [4, 1], [2, 3], [2, 3], [1, 4], [3, 2], [4, 1], [2, 3]]),
        "feedback": np.array([1, 2, 1, 2, 1, 1, 2, 1, 2, 1]),
        "observed": np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]),
    }
    experiment.append(ppt)

lower = [1e-10, 1e-10]
upper = [1, 10]
bounds = list(zip(lower, upper))

# FIXME: this is not working, metric is always overwritten
Fit = DifferentialEvolution(
    model = x, bounds = bounds, data = experiment, minimisation="LogLikelihood", mutation = 0.5, recombination = 0.7, strategy = "best1bin", tol = 0.001
)  # initialize the optimisation
Fit.minimise([0.60924690133760082, 5.6764639071806062])
Fit.optimise()

pp(Fit.fit)

fittings = [item.get('fun') for item in Fit.fit]
pp(fittings)
pp(Fit.parameters)

explore = Simulator(model = x, data = experiment, parameters = Fit.parameters)
explore.run()
explore.policies()
explore.generate()
pp(explore.generated)

explore.update(parameters = Fit.parameters[0])

explore.run()
explore.policies()
pp(explore.parameters)

recover = ParameterRecovery(
    model = explore, optimiser = "DifferentialEvolution", strategy = "grid",
    bounds = bounds, iteration = 2
    )
recover.recover()
# dimensions are: iteration, recovered/original, population (trial order)
alpha = recover.extract(key = 'alpha')

recovered = np.asarray(alpha[:, 0, :]).flatten()
original = np.asarray(alpha[:, 1, :]).flatten()


plt.scatter(recovered, original)
plt.show()

temperature = recover.extract(key = 'temperature')
recovered_temp = np.asarray(temperature[:, 0, :]).flatten()
original_temp = np.asarray(temperature[:, 1, :]).flatten()

plt.scatter(recovered_temp, original_temp)
plt.show()
