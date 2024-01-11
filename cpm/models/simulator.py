"""
Runs a simulation for each ppt in the data.
"""

import numpy as np
import pandas as pd

class Simulator():
    """
    A class representing a simulator for running simulations.
    """

    def __init__(self, function, data, params):
        """
        Initializes a Simulator object.

        Parameters:
        - function: The simulation function to be used.
        - data: The data required for the simulation.
        - params: The parameters required for the simulation.
        """
        self.function = function
        self.data = data
        self.parameters = params
        self.simulation = []

    def run(self):
        """
        Runs the simulation.

        Returns:
        - experiment: A list containing the results of the simulation.
        """
        for i in range(len(self.data)):
            sprint = self.function(self.data[i], self.parameters[i])
            sprint.run()
            output = sprint.export()
            output['ppt'] = self.data[i]['ppt']
            self.simulation.append(output)
        return None

    def probabilities(self):
        """
        Returns the probabilities from the simulation.

        Returns:
        - probabilities: A list containing the probabilities from the simulation.
        """
        probabilities = []
        for i in range(len(self.simulation)):
            probabilities.append(np.asarray(self.simulation[i]['probabilities']))
        return np.asarray(probabilities)