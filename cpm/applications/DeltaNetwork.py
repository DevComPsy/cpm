"""
This shows a complete least-mean-squares (LMS) or delta-rule adaptive network model, a
generalization of the Rescorla-Wagner learning rule.  It is a two-layer network with
one input layer and one output layer.  The input layer has one unit for each
stimulus, and the output layer has one unit for each response. The standard
Rescorla-Wagner model is a special case of this model.
"""

import numpy as np
import pandas as pd
from ..models.learning import DeltaRule
from ..models.decision import Softmax
from ..models.utils import Nominal


# FIXME: there were a lot of changes and I'm not sure if this is still correct
# NOTE: regardless of whether it works, this is representative of an implemented complete model
class DeltaNetwork:
    """
    Rescorla-Wagner model for learning in reinforcement learning tasks.

    Parameters
    ---------
    data : dict
        The input data for the model.
    params : dict
        The parameters for the model.

    Attributes:
        data (numpy.ndarray): The input data for the model.
        alpha (float): The learning rate parameter.
        temperature (float): The temperature parameter.
        weights (numpy.ndarray): The weights for each feature.
        policies (numpy.ndarray): The policies for each action, where first dimension is the trial and second dimension is the action.
        error (numpy.ndarray): The error for each trial. First dimension is the action, second dimension is the stimuli, and third dimension is the trial.
        lower (list): A named list. The lower bounds for the parameters.
        upper (list): A named list. The upper bounds for the parameters.
    """

    def __init__(self, data=None, params=None):

        # initialize the model parameters (defaults)
        self.weights = np.zeros((np.max(data["feedback"]), np.max(data["trials"])))
        self.alpha = 0.1
        self.temperature = 1

        # define the bounds for the parameters
        self.lower = [1e-10, 1e-10]
        self.upper = [1, 5]
        self.bounds = list(zip(self.lower, self.upper))
        self.priors = "normal"

        if params is not None:
            self.alpha = params["alpha"]
            self.temperature = params["temperature"]
            if "weights" in params.keys():
                self.weights = np.asarray(params["weights"])
            if "priors" in params.keys():
                self.priors = params["priors"]

        # initialize the data for the simulation
        self.training = data["trials"]
        self.feedback = data["feedback"]

        # initialize the output
        self.policies = np.zeros((data["trials"].shape[0], np.max(data["feedback"])))
        self.error = np.zeros(
            (self.weights.shape[0], self.weights.shape[1], data["trials"].shape[0])
        )

        # initialize some variables used in the implementation
        self.bits = np.max(data["trials"])  # the number of stimuli
        self.outcomes = np.max(self.feedback)  # the number of outcomes
        self.parameter_names = ["alpha", "temperature"]  # the names of the parameters

    def run(self):
        """
        Runs the model on the given data.

        This method iterates over each trial in the data and performs calculations
        using the weights and data values. It updates the policies and error
        arrays based on the calculations.

        Returns:
            None
        """
        for trial in range(
            self.training.shape[0],
        ):
            # FIXME: legacy code that needs to be updated
            active = Nominal(stimuli=self.training[trial], bits=self.bits)
            feedback = Nominal(stimuli=[self.feedback[trial]], bits=self.outcomes)
            current = active * self.weights
            self.policies[trial, :] = Softmax(self.temperature, current)
            self.weights = DeltaRule(self.alpha, current, feedback)
            self.error[:, :, trial] = change
        return None

    def export(self):
        """
        Exports the model results.

        Returns:
            dict: A dictionary containing the model results.
        """
        return {
            "name": "Rescorla-Wagner",
            "alpha": self.alpha,
            "temperature": self.temperature,
            "stimuli": self.training,
            "feedback": self.feedback,
            "policies": self.policies,
            "error": self.error,
            "weights": self.weights,
        }

    def reset(self, parameters):
        """
        Reset the model with new parameters.

        Parameters:
        - parameters (dict or list): The new parameters to set. If a dictionary is provided, the values are extracted
            based on the parameter names defined in the model. If a list is provided, it is assumed to be in the same
            order as the parameter names.

        Returns:
        - None
        """
        self.weights.fill(0)
        if isinstance(parameters, dict):
            self.weights = parameters.get("weights", self.weights)
            parameters_vector = [parameters.get(key) for key in self.parameter_names]
        else:
            parameters_vector = parameters
        self.alpha, self.temperature = parameters_vector[:2]

        return None

    def update_data(self, new=None):
        """
        Update the data in the model.

        Args:
            new (dict): A dictionary containing the new data.
                The dictionary should have the following keys:
                - 'trials': The updated trials data.
                - 'feedback': The updated feedback data.

        Returns:
            None
        """
        self.training = new["trials"]
        self.feedback = new["feedback"]
        return None
