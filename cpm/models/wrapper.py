import numpy as np
import copy
from prettyformatter import pprint as pp


class Wrapper:
    def __init__(self, model=None, data=None, parameters=None):
        ## set up environment to run the model
        ## FIXME: this is a hack, we need to find a better way to do this
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
        for i in range(len(self.training)):
            trial = {
                "input": np.asarray(self.training[i]),
                "feedback": np.asarray([self.feedback[i]]),
            }
            output = self.model(parameters=self.parameters, trial=trial)
            self.parameters.values = output.get("values")
            # output.compute(**trial)
            self.values = output.get('values').copy()
            self.simulation.append(output.copy())
            self.policies[i] = output.get('policy').copy()
        return None

    def reset(self, parameters=None):
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

    def summary(self):
        summary = {
            "values": self.values,
            "policies": self.policies,
            "model": self.model.summary(),
        }
        return summary

    def export(self):
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
