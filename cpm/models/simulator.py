"""
Runs a simulation for each ppt in the data.
"""
import numpy as np
import pandas as pd
import copy
import pickle as pkl

from .parameters import Parameters


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
        if isinstance(parameters, dict):
            self.parameters = [
                (copy.deepcopy(parameters)) for i in range(1, len(self.data) + 1)
            ]
        self.parameter_names = self.function.parameter_names
        self.simulation = []
        self.generated = []

    def run(self):
        """
        Runs the simulation.

        Returns:
        - experiment: A list containing the results of the simulation.
        """
        for i in range(len(self.data)):
            self.function.reset()
            evaluate = copy.deepcopy(self.function)
            evaluate.reset(parameters=self.parameters[i])
            evaluate.update_data(self.data[i])
            evaluate.run()
            output = copy.deepcopy(evaluate.export())
            output["ppt"] = copy.deepcopy(self.data[i]["ppt"])
            self.simulation.append(output)
            del evaluate, output
        return None

    def policies(self):
        """
        Return the trial- and participant-level information about the simulation.

        Returns
        ------
        policies : pandas.DataFrame
            A dataframe containing the policies from the simulation with the following columns:

            - ppt: The participant number.
            - policy_0, policy1, ...: The policy for each action, where each column is an action.
            - stimulus_0, stimulus1, ...: The stimuli present, where each column is a stimulus dimension.
        """
        policies = pd.DataFrame()
        for i in range(len(self.simulation)):
            policed = pd.DataFrame(self.simulation[i]["policies"])
            policed.columns = ["policy_" + str(col) for col in policed.columns]
            stimuli = pd.DataFrame(self.data[i].get("trials"))
            stimuli.columns = ["stimulus_" + str(col) for col in stimuli.columns]
            combined = pd.concat([policed, stimuli], axis=1)
            combined["ppt"] = self.simulation[i]["ppt"]
            policies = pd.concat([policies, combined], axis=0, ignore_index=True)
        return policies

    def update(self, parameters=None):
        """
        Updates the parameters of the simulation.

        Parameters:
        - params: The parameters to be updated.
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
            current = np.asarray(self.simulation[i]["policies"])
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
