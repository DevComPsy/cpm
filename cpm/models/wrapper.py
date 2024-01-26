import numpy as np
from ..components.utils import Nominal


class Wrapper:

    def __init__(self, model = None, data = None, params = None):
        ## set up environment to run the model
        ## FIXME: this is a hack, we need to find a better way to do this
        self.model = model
        self.data = data
        self.parameters = params.copy()
        self.weights = np.zeros((np.max(data["feedback"]), np.max(data["trials"])))
        if "weights" in self.parameters.keys():
            self.weights = self.parameters["weights"]
            params['weights'] = self.weights
        self.simulation = []
        self.outcomes = np.max(self.data["feedback"])
        self.bits = np.max(self.data["trials"])
        self.training = self.data["trials"]
        self.feedback = self.data["feedback"]
        self.magnitude = None
        self.continuous = False
        if "magnitude" in self.data:
            self.magnitude = self.data["magnitude"]
            self.continuous = True

        self.policies = np.zeros((self.data["trials"].shape[0], np.max(self.data["feedback"])))
        self.values = []
        self.parameter_names = list(params.keys())

    def run(self):
        for i in range(len(self.training)):
            # do something
            trial = {
                "weights": self.weights,
            }
            stimulus = {
                "target": self.training[i],
                "bits": self.bits,
            }
            trial["input"] = Nominal(**stimulus)
            feedback = {
                "target": np.asarray([self.feedback[i]]),
                "bits": self.outcomes,
            }
            if self.continuous:
                trial["magnitude"] = self.magnitude[i]
            trial["feedback"] = Nominal(**feedback)
            trial = {**trial, **self.parameters}
            output = self.model
            output.compute(**trial)
            self.weights = output.weights
            self.values.append(output.values)
            self.policies[i] = output.policy
        return None

    def reset(self, parameters = None):
        self.weights.fill(0)
        self.policies.fill(0)
        self.values = []
        if isinstance(parameters, dict):
            self.weights = parameters.get('weights', self.weights)
            self.parameters = parameters
        if isinstance(parameters, list):
            for keys in self.parameter_names:
                self.parameters[keys] = parameters[self.parameter_names.index(keys)]
        return None

    def update_data(self, new = None):
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
        self.training = new['trials']
        self.feedback = new['feedback']
        return None

    def summary(self):
        summary = {
            "weights": self.weights,
            "policies": self.policies,
            "model": self.model.summary(),
        }
        return summary

    def export(self):
        return {
            "weights": self.weights,
            "policies": self.policies,
            "name": self.model.name,
            **self.parameters,
        }


# arguments = {
#     "alpha": 0.1,
#     "temperature": 1,
#     "weights": np.array([[0.5, 0], [0, 0.5]]),
#     "input": np.array([1, 1]),
#     "teacher": np.array([1, 0]),
#     "attention": np.array([1, 0]),
#     "misc": np.array([1, 0]),
# }
