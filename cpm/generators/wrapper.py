import numpy as np
import pandas as pd
import copy
import pickle as pkl

## import local modules
from .parameters import Parameters, Value
from ..core.data import unpack_trials, determine_data_length
from ..core.exports import simulation_export


class Wrapper:
    """
    A `Wrapper` class for a model function in the CPM toolbox. It is designed to run a model for a **single** experiment (participant) and store the output in a format that can be used for further analysis.

    Parameters
    ----------
    model : function
        The model function that calculates the output(s) of the model for a single trial. See Notes for more information. See Notes for more information.
    data : pandas.DataFrame or dict
        If a `pandas.DataFrame`, it must contain information about each trial in the experiment that serves as an input to the model. Each trial is a complete row.
        If a `dictionary`, it must contains information about the each state in the environment or each trial in the experiment - all input to the model that are not parameters.
    parameters : Parameters object
        The parameters object for the model that contains all parameters (and initial states) for the model. See Notes on how it is updated internally.


    Returns
    -------
    Wrapper : object
        A Wrapper object.

    Notes
    -----
    The model function should take two arguments: `parameters` and `trial`. The `parameters` argument should be a [Parameter][cpm.generators.Parameters] object specifying the model parameters. The `trial` argument should be a dictionary or `pd.Series` containing all input to the model on a single trial. The model function should return a dictionary containing the model output for the trial. If the model is intended to be fitted to data, its output should contain the following keys:

    - 'dependent': Any dependent variables calculated by the model that will be used for the loss function.

    If a model output contains any keys that are also present in parameters, it updates those in the parameters based on the model output.

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
        # determine the number of trials
        self.__len__, self.__pandas__ = determine_data_length(data)

        self.dependent = []
        self.parameter_names = list(parameters.keys())

        self.__run__ = False
        self.__init_parameters__ = copy.deepcopy(parameters)

    def run(self):
        """
        Run the model.

        Returns
        -------
        None

        """
        for i in range(self.__len__):
            ## create input for the model
            trial = unpack_trials(self.data, i, self.__pandas__)
            ## run the model
            output = self.model(parameters=self.parameters, trial=trial)
            self.simulation.append(output.copy())

            ## update your dependent variables
            ## create dependent output on first iteration
            if i == 0:
                self.dependent = np.zeros(
                    (self.__len__, output.get("dependent").shape[0])
                )

            ## copy dependent variable from model output to attribute
            self.dependent[i] = np.asarray(output.get("dependent")).copy()

            ## update variables present in both parameters and model output
            self.parameters.update(
                **{
                    key: value
                    for key, value in output.items()
                    if key in self.parameters.keys()
                }
            )

        self.__run__ = True
        return None

    def reset(self, parameters=None, data=None):
        """
        Reset the model.

        Parameters
        ----------
        parameters : dict, array_like, pd.Series or Parameters, optional
            The parameters to reset the model with.

        Notes
        -----
        When resetting the model, and `parameters` is None, reset model to initial state.
        If parameter is `array_like`, it resets the only the parameters in the order they are provided,
        where the last parameter updated is the element in parameters corresponding to len(parameters).

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
            self.dependent.fill(0)
            self.simulation = []
            self.parameters = copy.deepcopy(self.__init_parameters__)
            self.__run__ = False
        # if dict, update using parameters update method
        if isinstance(parameters, dict) or isinstance(parameters, pd.Series):
            self.parameters.update(**parameters)
        # if list, update the parameters in for keys in range of 0:len(parameters)
        if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
            for keys in self.parameter_names[0 : len(parameters)]:
                value = parameters[self.parameter_names.index(keys)]
                self.parameters.update(**{keys: value})
        if data is not None:
            self.data = data
            self.__len__, self.__pandas__ = determine_data_length(data)
        return None

    def export(self):
        """
        Export the trial-level simulation details.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the model output for each participant and trial.
            If the output variable is organised as an array with more than one dimension, the output will be flattened.

        """
        return simulation_export([self.simulation])

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
