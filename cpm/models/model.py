# This file contains the model and generator classes for the model
import numpy as np

class Model:
    """
    A class representing a model.

    Parameters
    ----------
    functions : list
        A list of functions used in the model.

    Attributes
    ----------
    values : dict
        A dictionary containing the computed values of the model.
    policy : str
        The policy computed by the model.

    Methods
    -------
    compute(**kwargs)
        Computes the model based on the provided arguments for one single trial.

    """

    def __init__(self, functions=None, name=None):
        """
        Initialize a Model object.

        Parameters
        ----------
        functions : list, optional
            List of functions to be used in the model.
        name : str, optional
            Name of the model. If not provided, a default name will be assigned.

        Attributes
        ----------
        functions : list
            List of functions used in the model.
        values : None
            Placeholder for storing values.
        weights : None
            Placeholder for storing weights.
        policy : None
            Placeholder for storing policy.
        run : bool
            Flag indicating if the model has been run.
        name : str
            Name of the model.

        """
        self.functions = functions
        self.values = None
        self.weights = None
        self.policy = None
        self.run = False
        if name is not None:
            self.name = name
        else:
            self.name = "Model_" + str(len(functions))

    def compute(self, **kwargs):
        """
        Computes the model based on the provided arguments.

        Args:
            **kwargs: Keyword arguments used by the functions in the model. It must
            contain all the Named Arguments required by the functions supplied in the model.

        Returns:
            str: The computed policy.

        """
        input = kwargs.copy()
        outputs = {}
        x = None
        for function in self.functions:
            init = function(**input)
            name = init.config()['name']
            type = init.config()['type']
            if type == 'decision':
                init.compute()
                outputs[name] = init.policies
                self.policy = init.policies
            elif type == 'activation':
                init.compute()
                outputs[name] = init.weights
                input['activations'] = init.weights
            else:
                init.compute()
                x = init.weights.copy()
                outputs[name] = x
        self.values = outputs
        self.weights = x
        self.run = True
        return None

    def reset(self):
        """
        Resets the model to its initial state.

        """
        self.values = []
        self.policy = None

    def summary(self):
        """
        Prints a summary of the model.

        """
        if self.run == False:
            raise ValueError('The model has not been run yet.')
        else:
            summary = {}
            for function in self.functions:
                init = function()
                name = init.config()['name']
                type = init.config()['type']
                summary[type] = name
            summary['shape'] = self.weights.shape
            summary['variables'] = self.weights.shape[0] * self.weights.shape[1]
        return summary

    # TODO: config export
    # priors and bounds
    # bits
    # folder with data and model
    # config file and export options of what you are doing
    # TODO: reward magnitude (look at the two step task in brain explorer, the goblin heist task)
    # TODO: decision making task brain explorer