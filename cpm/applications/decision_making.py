import pandas as pd
import numpy as np
import warnings
import time
import cpm

from cpm.generators import Wrapper, Parameters, Value


class PUTM(Wrapper):

    def __init__(self, data=None, parameters_settings=None, generate=False):

        if parameters_settings is None:
            parameters_settings = [
                [0.5, 0, 2],
                [5, -50, 50],
                [10, -100, 100],
                [1, 0, 20],
                [0, 0, 1],
            ]
            warnings.warn("No parameters specified, using default parameters.")

        #  "alpha" : risk attitude
        # "kappa"  : gambling bias
        # "gamma"  : response repetition bias
        # "tau"    : choice temperature
        # "xi"     : irreducible noise

        # Define the maximal parameter set. All defined on the real line (-Inf, Inf),
        # to be transformed to suitably bounded range within the model function.
        parameters = {
            "alpha": Value(
                value=parameters_settings[0][0],
                lower=parameters_settings[0][1],
                upper=parameters_settings[0][2],
                prior="truncated_normal",
                args={"mean": 0.5, "sd": 0.25},
            ),
            "kappa": Value(
                value=parameters_settings[1][0],
                lower=parameters_settings[1][1],
                upper=parameters_settings[1][2],
                prior="truncated_normal",
                args={"mean": 5, "sd": 10},
            ),
            "gamma": Value(
                value=parameters_settings[2][0],
                lower=parameters_settings[2][1],
                upper=parameters_settings[2][2],
                prior="truncated_normal",
                args={"mean": 0, "sd": 20},
            ),
            "tau": Value(
                value=parameters_settings[3][0],
                lower=parameters_settings[3][1],
                upper=parameters_settings[3][2],
                prior="truncated_normal",
                args={"mean": 1, "sd": 2.5},
            ),
            "xi": Value(
                value=parameters_settings[4][0],
                lower=parameters_settings[4][1],
                upper=parameters_settings[4][2],
                prior="truncated_normal",
                args={"mean": 0, "sd": 0.5},
            ),
        }

        def model(parameters, trial):
            pass

        super().__init__(data=data, model=model, parameters=parameters)
