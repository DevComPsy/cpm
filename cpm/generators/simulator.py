"""
Runs a simulation for each ppt in the data.
"""

import numpy as np
import pandas as pd
import warnings
import copy
import pickle as pkl

from .parameters import Parameters, Value


class Simulator:
    """
    Initializes a Simulator object.

    Parameters
    ----------
    model : object
        The simulation function to be used.
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
    simulation : object
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
        if isinstance(parameters, dict):
            self.parameters = [
                (copy.deepcopy(Parameters(**parameters)))
                for i in range(1, len(self.data) + 1)
            ]
        self.parameter_names = self.function.parameter_names
        self.simulation = []
        self.generated = []

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
            evaluate.reset(parameters=self.parameters[i])
            evaluate.data = self.data[i]
            evaluate.run()
            output = copy.deepcopy(evaluate.export())
            self.simulation.append(output)
            del evaluate, output
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
        policies = pd.DataFrame()
        id = 0
        for i in self.simulation:
            ppt = pd.DataFrame()
            for k in i:
                row = pd.DataFrame()
                for key, value in k.items():
                    if len(list(np.array(value).shape)) > 1:
                        Warning(
                            f"Value of {key} is of shape {value.shape}. It should be 1D."
                        )
                    if isinstance(value, int) or isinstance(value, float):
                        value = np.array([value])
                    current = pd.DataFrame(value.flatten()).T
                    current.columns = [f"{key}_{i}" for i in range(current.shape[1])]
                    row = pd.concat([row, current], axis=1)
                ppt = pd.concat([ppt, row], axis=0)
            ppt["ppt"] = id
            id += 1
            policies = pd.concat([policies, ppt], axis=0)
        policies.reset_index()
        return policies

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
        results: array_like
            An array of dictionaries containing the results of the simulation.
        """
        for i in range(len(self.simulation)):
            current = np.asarray(self.simulation[i]["dependent"])
            self.generated.append({"observed": current})
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
