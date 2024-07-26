"""
Runs a simulation for each ppt in the data.
"""

import numpy as np
import pandas as pd
import copy
import pickle as pkl

from .parameters import Parameters
from ..core.data import unpack_participants
from ..core.generators import cast_parameters
from ..core.exports import simulation_export


class Simulator:
    """
    A `Simulator` class for a model in the CPM toolbox. It is designed to run a model for **multiple** participants and store the output in a format that can be used for further analysis.

    Parameters
    ----------
    wrapper : Wrapper
        An initialised Wrapper object for the model.
    data : pandas.core.groupby.generic.DataFrameGroupBy or list of dictionaries
        The data required for the simulation.
        If it is a pandas.core.groupby.generic.DataFrameGroupBy, as returned by `pandas.DataFrame.groupby()`, each group must contain the data (or environment) for a single participant.
        If it is a list of dictionaries, each dictionary must contain the data (or environment) for a single participant.
    parameters : Parameters, pd.DataFrame, pd.Series or list
        The parameters required for the simulation. It can be a Parameters object or a list of dictionaries whose length is equal to data. If it is a Parameters object, Simulator will use the same parameters for all simulations. It is a list of dictionaries, it will use match the parameters with data, so that for example parameters[6] will be used for the simulation of data[6].

    Returns
    -------
    simulator : Simulator
        A Simulator object.

    """

    def __init__(self, wrapper=None, data=None, parameters=None):
        self.wrapper = wrapper
        self.data = data

        self.groups = None
        self.__run__ = False
        self.__pandas__ = isinstance(data, pd.api.typing.DataFrameGroupBy)
        self.__parameter__pandas__ = isinstance(parameters, pd.DataFrame)
        if isinstance(self.__pandas__, pd.DataFrame):
            raise TypeError(
                "Data should be a pandas.DataFrameGroupBy object, not a pandas.DataFrame."
            )
        if self.__pandas__:
            self.groups = list(self.data.groups.keys())
        else:
            self.groups = np.arange(len(self.data))

        self.parameters = cast_parameters(parameters, len(self.groups))
        self.parameter_names = self.wrapper.parameter_names

        self.simulation = []
        self.generated = []

    def run(self):
        """
        Runs the simulation.

        Returns
        -------
        experiment: A list containing the results of the simulation.
        """

        for i in range(len(self.groups)):
            self.wrapper.reset()
            evaluate = copy.deepcopy(self.wrapper)
            ppt_data = unpack_participants(
                self.data, i, self.groups, pandas=self.__pandas__
            )
            ppt_parameter = unpack_participants(
                self.parameters, i, self.groups, pandas=self.__parameter__pandas__
            )
            evaluate.reset(parameters=ppt_parameter, data=ppt_data)
            evaluate.run()
            output = copy.deepcopy(evaluate.simulation)
            self.simulation.append(output.copy())
            del evaluate, output

        self.simulation = np.array(self.simulation, dtype=object)
        self.__run__ = True
        return None

    def export(self):
        """
        Return the trial- and participant-level information about the simulation.

        Returns
        ------
        policies : pandas.DataFrame
            A dataframe containing the the model output for each participant and trial.
            If the output variable is organised as an array with more than one dimension, the output will be flattened.
        """
        return simulation_export(self.simulation)

    def update(self, parameters=None):
        """
        Updates the parameters of the simulation.

        Parameters
        ----------
        parameters : object
            The parameters to be updated.
        """
        if isinstance(parameters, Parameters):
            raise TypeError("Parameters must be a dictionary or array_like.")
        ## if parameters is a single set of parameters, then repeat for each ppt
        if isinstance(parameters, dict):
            self.parameters = [
                (copy.deepcopy(parameters)) for i in range(1, len(self.data) + 1)
            ]
        if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
            self.parameters = parameters
        return None

    def generate(self, variable="dependent"):
        """
        Generate data for parameter recovery, etc.

        Parameters
        ----------
        variable: str
            Name of the variable to pull out from model output.

        Returns
        ------
        results: numpy.ndarray
            An array of dictionaries containing the results of the simulation.
        """
        append = []
        for ppt in self.simulation:
            one = {"observed": np.zeros((self.wrapper.__len__, 1))}
            for k in range(self.wrapper.__len__):
                one.get("observed")[k] = np.array([ppt[k].get(variable)])
            append.append(one)
        self.generated = copy.deepcopy(append)
        return None

    def reset(self):
        """
        Resets the simulation.
        """
        self.simulation = []
        self.generated = []
        return None

    def save(self, filename=None):
        """
        Saves the simulation results.

        Parameters
        ----------
        filename : str
            The name of the file to save the results to.
        """
        if filename is None:
            filename = "simulation"
        pkl.dump(self, open(filename + ".pkl", "wb"))
        return None
