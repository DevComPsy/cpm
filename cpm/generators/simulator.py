"""
Runs a simulation for each ppt in the data.
"""

import numpy as np
import pandas as pd
import warnings
import copy
import pickle as pkl

from .parameters import Parameters, Value
from .utils import simulation_export


class Simulator:
    """
    A `Simulator` class for a model in the CPM toolbox. It is designed to run a model for **multiple** participants and store the output in a format that can be used for further analysis.

    Parameters
    ----------
    model : Wrapper
        An initialised Wrapper object for the model.
    data : object
        The data required for the simulation.
    parameters : object
        The parameters required for the simulation.

    Attributes
    ----------
    function : object
        The simulation function to be used.
    data : object
        The data required for the simulation.
    parameters : object
        The parameters required for the simulation.
    parameter_names : object
        The names of the parameters.
    simulation : numpy.ndarray
        The results of the simulation, including the policies and the states.
    generated : object
        The results of the simulation, only including the policies.

    Returns
    -------
    simulator : Simulator
        A Simulator object.

    """

    def __init__(self, model=None, data=None, parameters=None):
        self.function = model
        self.data = data
        self.parameters = copy.deepcopy(parameters)
        if len(self.data) != len(self.parameters):
            self.parameters = []
            self.parameters = [
                copy.deepcopy(parameters) for i in range(1, len(self.data) + 1)
            ]
            warnings.warn(
                "The number of parameter sets and number of participants in data do not match.\nUsing the same parameters for all participants."
            )
        self.parameter_names = self.function.parameter_names
        self.simulation = []
        self.generated = []
        self.__run__ = False

    def run(self):
        """
        Runs the simulation.

        Returns
        -------
        experiment: A list containing the results of the simulation.
        """
        for i in range(len(self.data)):
            self.function.reset()
            evaluate = copy.deepcopy(self.function)
            evaluate.data = self.data[i]
            evaluate.reset(parameters=self.parameters[i])
            evaluate.run()
            output = copy.deepcopy(evaluate.simulation)
            self.simulation.append(output.copy())
            del evaluate, output

        self.simulation = np.array(self.simulation)
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
            TypeError("Parameters must be a dictionary or array_like.")
        ## if parameters is a single set of parameters, then repeat for each ppt
        if isinstance(parameters, dict):
            self.parameters = [
                (copy.deepcopy(parameters)) for i in range(1, len(self.data) + 1)
            ]
        if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
            self.parameters = parameters
        return None

    def generate(self):
        """
        Generate data for parameter recovery, etc.

        Returns
        ------
        results: numpy.ndarray
            An array of dictionaries containing the results of the simulation.
        """
        for i in self.simulation:
            current = []
            for j in i:
                current.append(j.get("dependent"))
            self.generated.append({"observed": np.asarray(current)})
        self.generated = np.array(self.generated)
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
