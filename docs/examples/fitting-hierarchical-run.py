import numpy
import pandas as pd
import cpm
import cpm.datasets as datasets
from cpm.generators import Parameters, Value
import functions as f

import warnings

data = datasets.load_bandit_data()
data.head()
data["observed"] = data["response"]  # We need to specify the observed variable for the fitting
## we ignore the first output, because it is the model that generates the data
_, model_delta, parameters = f.model_delta(data)

warnings.filterwarnings("ignore", category=RuntimeWarning)
fit = cpm.optimisation.FminBound(
    model=model_delta,  # Wrapper class with the model we specified from before
    data=data.groupby('ppt'),  # the data as a list of dictionaries
    minimisation=cpm.optimisation.minimise.LogLikelihood.bernoulli,
    parallel=True,
    prior=False,
    ppt_identifier="ppt",
    display=False,
    number_of_starts=5,
    # everything below is optional and passed directly to the scipy implementation of the optimiser
    approx_grad=True
)

from cpm.hierarchical import VariationalBayes

variational = VariationalBayes(
    optimiser=fit,
    iteration=40,
    chain=4,
    tol=1e-3,
    convergence="lme",
    quiet=False,
)

variational.optimise()
variational.hyperparameters
variational.hyperparameters.to_csv("04-fitting-hierarchical-hyperparameters.csv")