"""
Runs a simulation for each ppt in the data.
"""

import numpy as np
import pandas as pd
import copy

class Simulator():
    """
    A class representing a simulator for running simulations.
    """

    def __init__(self, model = None, data = None, parameters = None):
        """
        Initializes a Simulator object.

        Parameters:
        - function: The simulation function to be used.
        - data: The data required for the simulation.
        - params: The parameters required for the simulation.
        """
        self.function = model
        self.data = data
        self.parameters = parameters
        if len(parameters) == 1:
            self.parameters = [(i, parameters.copy()) for i in range(1, len(self.data)+1)]
        self.parameter_names = self.function.parameter_names
        self.simulation =[]
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
            evaluate.reset(self.parameters[i])
            evaluate.update_data(self.data[i])
            evaluate.run()
            output = copy.deepcopy(evaluate.export())
            output['ppt'] = copy.deepcopy(self.data[i]['ppt'])
            self.simulation.append(output)
            del evaluate, output
        return None

    def policies(self):
        """
        Returns the policies from the simulation.

        Returns:
        - policies: A list containing the policies from the simulation.
        """
        policies = []
        for i in range(len(self.simulation)):
            policies.append(np.asarray(self.simulation[i]['policies']))
        return np.asarray(policies)

    def update(self, parameters = None):
        """
        Updates the parameters of the simulation.

        Parameters:
        - params: The parameters to be updated.
        """
        self.parameters = parameters
        ## if parameters is a single set of parameters, then repeat for each ppt
        if isinstance(parameters, dict):
            self.parameters = [(parameters.copy()) for i in range(1, len(self.data)+1)]
        return None

    def generate(self):
        """
        Exports the results of the simulation.

        Returns:
        - results: A list containing the results of the simulation.
        """
        for i in range(len(self.simulation)):
            current = np.asarray(self.simulation[i]['policies'])
            self.generated.append({'observed': current})
        return None

    def reset(self):
        """
        Resets the simulation.
        """
        self.simulation = []
        self.generated = []
        return None