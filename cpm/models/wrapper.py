import numpy as np
import copy
from prettyformatter import pprint as pp


class Wrapper:
    """
    A wrapper class for a model in the CPM toolbox.

    Parameters
    ----------
    model : function
        The model function that calculates the output(s) of the model for a single trial.
    data : dict
        A dictionary containing the data for the model. For the correct format of the dictionary, see the documentation of the TBA.
    parameters : [Parameters][cpm.models.Parameters] object
        The parameters object for the model that contains all parameters for the model.

    Attributes
    ----------
    model : object
        The model object.
    data : dict
        The data for the model.
    parameters : object
        The parameters object for the model.
    values : ndarray
        The values array.
    simulation : list
        The list of simulation results.
    training : ndarray
        The training data.
    feedback : ndarray
        The feedback data.
    policies : ndarray
        The policies array.
    parameter_names : list
        The list of parameter names.

    Returns
    -------
    Wrapper : object
        A Wrapper object.
    """

    def __init__(self, model=None, data=None, parameters=None):
        self.model = model
        self.data = data
        self.parameters = copy.deepcopy(parameters)
        self.values = np.zeros((np.max(data["feedback"]), np.max(data["trials"])))
        if "values" in self.parameters.__dict__.keys():
            self.values = self.parameters.values
        self.simulation = []
        self.training = self.data["trials"]
        self.feedback = self.data["feedback"]

        self.policies = np.zeros(
            (self.data["trials"].shape[0], np.max(self.data["feedback"]))
        )
        self.parameter_names = list(parameters.__dict__.keys())

    def run(self):
        """
        Run the model.

        Returns
        -------
        None

        """
        for i in range(len(self.training)):
            trial = {
                "input": np.asarray(self.training[i]),
                "feedback": np.asarray([self.feedback[i]]),
            }
            output = self.model(parameters=self.parameters, trial=trial)
            self.parameters.values = output.get("values")
            # output.compute(**trial)
            self.values = output.get("values").copy()
            self.simulation.append(output.copy())
            self.policies[i] = output.get("policy").copy()
        return None

    def reset(self, parameters=None):
        """
        Reset the model.

        Parameters
        ----------
        parameters : dict or list, optional
            The parameters to reset the model with.

        Returns
        -------
        None

        """
        self.values.fill(0)
        self.policies.fill(0)
        self.values = []
        if isinstance(parameters, dict):
            self.values = parameters.get("values", self.values)
            self.parameters = parameters
        if isinstance(parameters, list):
            for keys in self.parameter_names:
                self.parameters[keys] = parameters[self.parameter_names.index(keys)]
        return None

    def update_data(self, new=None):
        """
        Update the data in the model.

        Parameters
        ----------
        new : dict
            A dictionary containing the new data.
            The dictionary should have the following keys:
            
            - 'trials': The updated trials data.
            - 'feedback': The updated feedback data.

        Returns
        -------
        None

        """
        self.training = new["trials"]
        self.feedback = new["feedback"]
        return None

    def summary(self):
        """
        Get a summary of the model.

        Returns
        -------
        dict
            A dictionary containing the model summary.

            - 'values': The values array.
            - 'policies': The policies array.
            - 'model': The model summary.

        """
        summary = {
            "values": self.values,
            "policies": self.policies,
        }
        return summary

    def export(self):
        """
        Export the model configurations.

        Returns
        -------
        dict
            A dictionary containing the exported model configurations.

            - 'values': The values array.
            - 'policies': The policies array.
            - 'name': The name of the model.
            - 'parameters': The parameters of the model.

        """
        return {
            "values": self.values,
            "policies": self.policies,
            "name": self.model.name,
            **self.parameters,
        }


# arguments = {
#     "alpha": 0.1,
#     "temperature": 1,
#     "values": np.array([[0.5, 0], [0, 0.5]]),
#     "input": np.array([1, 1]),
#     "teacher": np.array([1, 0]),
#     "attention": np.array([1, 0]),
#     "misc": np.array([1, 0]),
# }
