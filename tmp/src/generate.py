from cpm.generators import Simulator, Parameters, Value, Wrapper
from cpm.utils import dict_to_pandas, pandas_to_dict
import pandas as pd
import numpy as np


def generate_data(data, model, mean=[0.25, 5], sd=[0.25, 2.5]):
    """The function `generate_data` simulates data based on specified parameters and
    model, and then exports the simulated responses.

    Parameters
    ----------
    dta: pd.DataFrame
        The data used for the simulation.
    model: cpm.generators.Wrapper
        The `generate_data` function you provided seems to be generating simulated data
    based on the specified parameters and model. However, it looks like the
    description of the `model` parameter is missing. Could you please provide more
    information about the `model` parameter so that I can assist you further with
    understanding the
    experiment: list
        A list of dictionaries containing the data used for the simulation. See cpm.generator.Wrapper for more information on the format of the data.
    mean: array-like, optional
        The `mean` parameter in the `generate_data` function represents the mean values
    used for generating the parameters `alpha` and `beta`.
    sd: array-like, optional
        The `sd` parameter in the `generate_data` function represents the standard
    deviation for the truncated normal distribution used as a prior for sampling the
    parameters `alpha` and `beta`.

    Returns
    -------
    dta: pd.DataFrame
        The simulated data with responses based on the specified parameters and model.
    """

    parameters = Parameters(
        # freely varying parameters are indicated by specifying priors
        alpha=Value(
            value=0.5,
            lower=1e-2,
            upper=1,
            prior="truncated_normal",
            args={"mean": mean[0], "sd": sd[0]},
        ),
        beta=Value(
            value=2,
            lower=0,
            upper=10,
            prior="truncated_normal",
            args={"mean": mean[1], "sd": sd[1]},
        ),
        values=np.ones(4) / 4,
    )

    experiment = pandas_to_dict(
        data,
        participant="ppt",
        stimuli="stimulus",
        feedback="reward",
        observed="responses",
        trial_number="trial",
    )

    pp = parameters.sample(len(data.ppt.unique()))

    sim = Simulator(wrapper=model, data=experiment, parameters=pp)
    sim.run()
    dat = sim.export()
    responses = dat.response_0.reset_index()
    data.responses = responses.response_0.astype(int)

    return data
