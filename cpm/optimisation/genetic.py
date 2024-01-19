from scipy.optimize import differential_evolution
from . import minimise
from . import utils

class DifferentialEvolution:
    def __init__(self, model = None, bounds = None, data = None, minimisation="LogLikelihood", **kwargs):
        self.function = model
        self.data = data
        self.loss = getattr(minimise, minimisation)
        self.kwargs = kwargs
        self.fit = []
        self.parameters = []
        self.participant = data[0]
        self.parameter_names = self.function.parameter_names
        if hasattr(self.function, 'bounds'):
            self.bounds = self.function.bounds
        else:
            self.bounds = bounds
            # raise ValueError("You must define the parameter bounds in the Model object.")

    def minimise(self, parameters, **args):
        evaluate = self.function
        evaluate.reset(parameters.copy())
        evaluate.update_data(self.participant)
        evaluate.run()
        predicted = evaluate.policies
        observed = self.participant["observed"]
        metric = self.loss(predicted, observed)
        out = metric.copy() # FIXME: testing weird memory pointer issue
        del evaluate, predicted, observed, metric
        return out

    def optimise(self):
        for i in range(len(self.data)):
            self.participant = self.data[i]
            result = differential_evolution(self.minimise, self.bounds, **self.kwargs)
            result["ppt"] = self.participant["ppt"]
            # add the parameters to the list
            parameters = utils.ExtractParamsFromFit(
                data=result["x"], keys=self.parameter_names
            )
            self.parameters.append(parameters.copy())
            # add the results to the list
            self.fit.append(utils.ConvertToDict(result.copy()))
            del result, parameters
        return None
