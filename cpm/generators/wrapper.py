import numpy as np
import pandas as pd
import copy
import pickle as pkl
from scipy.stats import norm

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

    Information on how to compile the model can be found in the [Tutorials - Build your model][/tutorials/defining-model] module.
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



class MetaSignalDetectionWrapper:

    def __init__(self, model=None, data=None, parameters=None):
        self.model = model
        self.data = data
        self.__init_parameters__ = copy.deepcopy(parameters)

        # check if parameters is provided as a dictionary with participants as keys
        if not isinstance(parameters, Parameters):
            assert isinstance(parameters, dict), "Parameters must be a dictionary with participants as keys if not a Parameters object."
            assert all([isinstance(p, Parameters) for p in parameters.values()]), "All entries in the dictionary must be Parameters objects."
            self.multiple_ppts = True
            first_ppt = list(parameters.keys())[0]
            self.ppt = first_ppt
        else:
            self.multiple_ppts = False
            self.ppt = None
        self.parameters = parameters
        
        # self.values = np.zeros(1)
        # if "values" in self.parameters.__dict__.keys():
        #     self.values = self.parameters.values
        self.simulation = None
        self.data = data
        # determine the number of trials
        self.__len__, self.__pandas__ = determine_data_length(data)

        self.dependent = {} if self.multiple_ppts else None
        self.parameter_names = list(self.parameters[self.ppt].keys()) if self.multiple_ppts else list(self.parameters.keys())
        self.parameter_sizes = {
            key: self.parameters[self.ppt][key].value.size if isinstance(self.parameters[self.ppt][key].value, np.ndarray) else 1 
            for key in self.parameters[self.ppt].keys()
            }

        self.__run__ = False
    
    def run(self):
        """
        Run the model.

        Returns
        -------
        None

        """
        ## run the model
        output = self.model(parameters=self.parameters[self.ppt]) if self.multiple_ppts else self.model(parameters=self.parameters)
        self.simulation = output.copy()

        ## update your dependent variables
        if self.multiple_ppts:
            self.dependent[self.ppt] = output.get("dependent").copy()
        else:
            self.dependent = output.get("dependent").copy()

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
    
    def reset(self, parameters=None, data=None, ppt=None):
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
        self.ppt = ppt if ppt is not None else self.ppt
        if self.__run__:
            self.dependent = {} if self.multiple_ppts else None
            self.simulation = None
            self.parameters = self.__init_parameters__
            self.__run__ = False
        if isinstance(parameters, Parameters):
            self.parameters[self.ppt] = parameters
        # if dict, update using parameters update method
        if isinstance(parameters, dict):
            if isinstance(parameters[self.ppt], Parameters):
                self.parameters[self.ppt].update(**parameters[self.ppt])
            else:
                raise NotImplementedError("Dictionary update not implemented for MetaSignalDetectionWrapper")
            if isinstance(parameters, pd.Series):
                raise NotImplementedError("Series update not implemented for MetaSignalDetectionWrapper")
        # if list, update the parameters in for keys in range of 0:len(parameters)
        if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
            offset = 0
            for idx, keys in enumerate(self.parameter_names):
                if self.parameter_sizes[keys] > 1:
                    value = parameters[self.parameter_names.index(keys)+offset:self.parameter_names.index(keys)+offset+self.parameter_sizes[keys]]
                    offset += self.parameter_sizes[keys] - 1
                else:
                    value = parameters[self.parameter_names.index(keys)+offset]
                self.parameters[self.ppt].update(**{keys: value}) if self.multiple_ppts else self.parameters.update(**{keys: value})
                if idx + offset + 1 == len(parameters):
                    break
        if data is not None:
            self.data = data
            self.__len__, self.__pandas__ = determine_data_length(data)
        return None
    
    def sample_ppt(self, num_trials = None, num_samples=100, ppt=None):

        assert ppt is not None, "ppt must be provided for MetaSignalDetectionWrapper"

        if num_trials is not None:
            assert NotImplementedError("num_trials argument not implemented for MetaSignalDetectionWrapper")

        stimulus = self.data['stimulus'].values
        stimulus = np.random.randint(2, size=num_trials)

        d1 = self.parameters[ppt].d1
        t1c1 = self.parameters[ppt].t1c1
        meta_d1 = self.parameters[ppt].meta_d1.value
        t2c1 = self.parameters[ppt].t2c1.value
        nbins = self.parameters[ppt].nbins

        t2c1 = np.concatenate([
            [-np.inf],
            t2c1[:nbins-1],
            [0],
            t2c1[nbins-1:],
            [np.inf],
        ])

        constant_criterion = t1c1 * meta_d1 / d1
        mu_S1 = -meta_d1 / 2 - constant_criterion
        mu_S2 = meta_d1 / 2 - constant_criterion

        s_S1 = norm.rvs(mu_S1, 1, size=(num_samples, (stimulus == 0).sum()))
        s_S2 = norm.rvs(mu_S2, 1, size=(num_samples, (stimulus == 1).sum()))

        nR_S1 = np.zeros((num_samples, 2*nbins))
        nR_S2 = np.zeros((num_samples, 2*nbins))

        for i in range(num_samples):
            nR_S1[i] = np.histogram(s_S1[i], bins=t2c1)[0]
            nR_S2[i] = np.histogram(s_S2[i], bins=t2c1)[0]
        
        return nR_S1, nR_S2





    