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
        The model function that calculates the output(s) of the model for a single trial. See Notes for more information.
    data : dict
        A dictionary containing the data for the model. The data for the model. This is a dictionary that contains information about the each state or trial in the environment or the experiment.
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
    data : dict
        The data for the model. This is a dictionary that contains information about the each state or trial in the environment or the experiment.
    policies : ndarray
        The policies array.
    parameter_names : list
        The list of parameter names.

    Returns
    -------
    Wrapper : object
        A Wrapper object.

    Notes
    -----
    The model function should take two arguments: `parameters` and `trial`. The `parameters` argument should be a [Parameter][cpm.generators.Parameters] object specifying the model parameters. The `trial` argument should be a dictionary containing the data for a single trial. The model function should return a dictionary containing the model output for the trial. The model output should contain the following keys:

    - 'values': The values array.
    - 'policy': The policies array.
    - 'dependent': Any dependent variables calculated by the model that will be used for the loss function.
    - 'other': Any other output from the model.
    """

    def __init__(self, model=None, data=None, parameters=None):
        self.model = model
        self.data = data
        self.parameters = copy.deepcopy(parameters)
        self.values = np.zeros(1)
        if "values" in self.parameters.__dict__.keys():
            self.values = self.parameters.values
        self.simulation = []
        self.data = data

        self.shape = [(np.array(v).shape) for k, v in self.data.items() if k != "ppt"]
        self.__len__ = np.max(self.shape)
        self.dependent = []
        self.parameter_names = list(parameters.keys())

        self.__run__ = False

    def run(self):
        """
        Run the model.

        Returns
        -------
        None

        """
        for i in range(self.__len__):
            ## create input for the model
            trial = {k: self.data[k][i] for k in self.data.keys() if k != "ppt"}
            ## run the model
            output = self.model(parameters=self.parameters, trial=trial)
            self.simulation.append(output.copy())
            self.parameters.values = Value(output.get("values"))

            if i == 0:
                self.dependent = np.zeros(
                    (self.__len__, output.get("dependent").shape[0])
                )

            self.dependent[i] = np.asarray(output.get("dependent")).copy()
        self.values = output.get("values").copy()
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
            self.dependent.fill(0)
            self.parameters.values = self.values
            self.__run__ = False
        # if dict, update using parameters update method
        if isinstance(parameters, dict):
            self.parameters.update(**parameters)
        # if list, update the parameters in for keys in range of 0:len(parameters)
        if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
            for keys in self.parameter_names[0 : len(parameters)]:
                value = parameters[self.parameter_names.index(keys)]
                setattr(self.parameters, keys, value)
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
            **self.parameters.export(),
        }
        return summary

    def export(self):
        """
        Export the model configurations.

        Returns
        -------
        list
            A list containing model output for each trial.

        """
        return self.simulation

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
