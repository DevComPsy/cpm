# This file contains the model and generator classes for the model
import numpy as np

class Model:
    """
    A class representing a model.

    Attributes:
        functions (list): A list of functions used in the model.
        values (dict): A dictionary containing the computed values of the model.
        policy (str): The policy computed by the model.

    Methods:
        compute: Computes the model based on the provided arguments for one single trial.
    """

    def __init__(self, functions=None):
        self.functions = functions
        self.values = []
        self.policy = None

    def compute(self, **kwargs):
        """
        Computes the model based on the provided arguments.

        Args:
            **kwargs: Keyword arguments used by the functions in the model. It must
            contain all the Named Arguments required by the functions supplied in the model.

        Returns:
            str: The computed policy.

        """
        outputs = {}
        x = None
        policy = None
        for function in self.functions:
            init = function(**kwargs)
            name = init.config()['name']
            type = init.config()['type']
            if type == 'decision':
                init.compute()
                outputs[name] = init.policies
                policy = init.policies
                print(policy)
                print(init.policies)
            else:
                init.compute()
                x = init.weights
                outputs[name] = x
        self.values = outputs
        return policy


# arguments = {
#     'alpha' : 0.1,
#     'temperature': 1,
#     'weights' : np.array([[0.5, 0], [0, 0.5]]),
#     'input' : np.array([1, 1]),
#     'teacher' : np.array([1, 0]),
#     'attention' : np.array([1, 0]),
#     'misc' : np.array([1, 0])
#     }

# input = arguments


# mine = Model(functions = [LinearActivation, Softmax, DeltaRule])
# mine.compute(**input)

    # def get_config(self):
    #     return None