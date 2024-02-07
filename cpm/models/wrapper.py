import numpy as np
import pandas as pd
import copy
import pickle as pkl

## import local modules
from .parameters import Parameters, Value


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

        self.__run__ = False

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
            self.policies[i] = np.asarray(output.get("policy")).copy()
        self.__run__ = True
        return None

    def reset(self, parameters=None):
        """
        Reset the model.

        Parameters
        ----------
        parameters : dict or array_like, optional
            The parameters to reset the model with.

        Notes
        -----
        When resetting the model, the values and policies arrays are reset to zero.
        If values are provided by the user, the values array is updated with the new values.

        Examples
        --------
        >>> x = Wrapper(model = mine, data = data, parameters = params)
        >>> x.run()
        >>> x.reset(parameters = [0.1, 1])
        >>> x.run()
        >>> x.reset(parameters = {'alpha': 0.1, 'temperature': 1})
        >>> x.run()
        >>> x.reset(parameters = np.array([0.1, 1, 0.5]))
        >>> x.run()

        Returns
        -------
        None

        """
        if self.__run__:
            self.values.fill(0)
            self.policies.fill(0)
            self.parameters.values = self.values
            self.__run__ = False
        if isinstance(parameters, dict):
            if "values" not in parameters.keys():
                updates = Parameters(**parameters, values=self.values)
            else:
                updates = Parameters(**parameters)
            self.parameters = copy.deepcopy(updates)
        if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
            for keys in self.parameter_names[0 : len(parameters)]:
                value = parameters[self.parameter_names.index(keys)]
                setattr(self.parameters, keys, value)
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
        export = {
            "values": self.values,
            "policies": self.policies,
            **self.parameters.export(),
        }
        return export

    def save(self, filename=None):
        """
        Save the model.

        Parameters
        ----------
        filename : str
            The name of the file to save the results to.

        Returns
        -------
        None

        Examples
        --------
        >>> x = Wrapper(model = mine, data = data, parameters = params)
        >>> x.run()
        >>> x.save('simulation')

        If you wish to save a file in a specific folder, provide the relative path.

        >>> x.save('results/simulation')
        >>> x.save('../archives/results/simulation')
        """
        if filename is None:
            filename = "simulation"
        pkl.dump(self, open(filename + ".pkl", "wb"))
        return None
