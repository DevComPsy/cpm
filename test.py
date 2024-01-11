from cpm.models import RescorlaWagner, Simulator  # import model
from cpm.optimisation import DifferentialEvolution  # import optimisation
from cpm.optimisation import utils  # import utils
from prettyformatter import pprint
import numpy as np
import pandas as pd

data = {
    "trials": np.array([[2, 3], [1, 4], [2, 3], [1, 4], [2, 3]]),
    "feedback": np.array([1, 2, 1, 2, 1]),
    "observed": np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]),
}

pp.pprint(data)

params = {  # define the parameters
    "alpha": 0.25,
    "temperature": 5,
    'weights': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
}

RW = RescorlaWagner(data, params) # initialize the model
RW.run()  # run the model
RW.policies  # investigate TODO: rename to policies

pprint(RW.export())  # export the results


# run the simulator on multiple ppt

## set up multiple people
experiment = []
for i in range(100):
    ppt = {
        "ppt": i,
        "trials": np.array([[2, 3], [1, 4], [3, 2], [4, 1], [2, 3]]),
        "feedback": np.array([1, 2, 1, 2, 1]),
        "observed": np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]),
    }
    experiment.append(ppt)

## set up multiple parameters
parameters = []
for i in range(len(experiment)):
    params = {
        "alpha": np.random.uniform(0, 1),
        "temperature": np.random.uniform(1, 5),
    }
    parameters.append(params)

Simulation = Simulator(RescorlaWagner, experiment, parameters)
Simulation.run()
pprint(Simulation.simulation)
pprint(Simulation.probabilities())

pro = Simulation.probabilities()
utils.ConvertToPandas(pro)

Optimise = DifferentialEvolution(
    RescorlaWagner, experiment, minimisation="LogLikelihood"
)  # initialize the optimisation
Optimise.minimise([0.5, 2])

Optimise.optimise()

pprint(Optimise.fit)

pd.DataFrame(Optimise.fit)