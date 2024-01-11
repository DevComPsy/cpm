from scipy.optimize import differential_evolution
from . import minimise
from . import utils

class DifferentialEvolution:
    def __init__(self, function, data, minimisation="LogLikelihood", **kwargs):
        self.function = function
        self.data = data
        self.loss = getattr(minimise, minimisation)
        self.kwargs = kwargs
        self.fit = []
        self.parameters = []
        self.participant = data[0]
        self.bounds = self.function(data[0]).bounds
        self.parameter_names = self.function(data[0]).parameter_names

    def minimise(self, parameters):
        evaluate = self.function(data=self.participant)
        evaluate.reset(parameters)
        evaluate.run()
        predicted = evaluate.probabilities
        observed = self.participant["observed"]
        metric = self.loss(predicted, observed)
        return metric

    def optimise(self):
        for i in range(len(self.data)):
            self.participant = self.data[i]
            result = differential_evolution(self.minimise, self.bounds, **self.kwargs)
            result["ppt"] = self.participant["ppt"]
            # add the parameters to the list
            parameters = utils.ExtractParamsFromFit(
                data=result["x"], keys=self.parameter_names
            )
            self.parameters.append(parameters)
            # add the results to the list
            self.fit.append(utils.ConvertToDict(result))
        return None
