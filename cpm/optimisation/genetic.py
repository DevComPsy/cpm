from scipy.optimize import differential_evolution
import copy
from . import minimise
from . import utils
from ..models import Simulator, Wrapper

class DifferentialEvolution:
    def __init__(self, model = None, bounds = None, data = None, minimisation="LogLikelihood", **kwargs):
        self.function = model
        self.data = data
        self.loss = getattr(minimise, minimisation)
        self.kwargs = kwargs
        self.fit = []
        self.parameters = []
        self.participant = data[0]
        if isinstance(model, Wrapper):
            self.parameter_names = self.function.parameter_names
        if isinstance(model, Simulator):
            self.parameter_names = self.function.function.parameter_names
        if hasattr(self.function, 'bounds'):
            self.bounds = self.function.bounds
        else:
            self.bounds = bounds
            # raise ValueError("You must define the parameter bounds in the Model object.")

    def minimise(self, pars, **args):
        self.function.reset()  
        self.function.parameters = {'alpha': pars[0], 'temperature': pars[1]}
        self.function.run()
        predicted = self.function.policies
        observed = self.participant.get("observed")
        metric = self.loss(predicted, observed)
        return metric

    def optimise(self):
        for i in range(len(self.data)):
            self.participant = self.data[i]
            self.function.update_data(self.participant)
            objective = copy.deepcopy(self.minimise)
            result = differential_evolution(objective, self.bounds, **self.kwargs)
            # add the parameters to the list
            fitted_parameters = utils.ExtractParamsFromFit(
                data=result.x, keys=self.parameter_names
            )
            self.parameters.append(fitted_parameters.copy())
            # add the results to the list
            self.fit.append({"parameters": result.x,
                             "fun": result.fun})
            del result, fitted_parameters, objective
        return None

    def reset(self):
        self.fit = []
        self.parameters = []
        return None