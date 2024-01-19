import numpy as np
from .. import optimisation
from ..optimisation import utils
from ..models import Simulator
from . import strategies

class ParameterRecovery:

    def __init__(self, model = None, optimiser = None, strategy = None, iteration = 1000, **kwargs):
        self.function = model
        self.template = self.function.parameters[0]
        self.optimisation = getattr(optimisation, optimiser)
        self.data = self.function.data
        self.strategy = getattr(strategies, strategy)
        self.iteration = iteration
        self.population = len(self.data)
        self.kwargs = kwargs
        self.output = []

    def recover(self):
        for step in range(self.iteration):
            parameters = self.strategy(template = self.template, population = self.population, **self.kwargs)
            self.function.update(parameters = parameters)
            self.function.run()
            self.function.generate()
            data = self.function.generated.copy()
            for i in range(len(self.data)):
                self.data[i]['observed'] = data[i]['observed']
            optim = self.optimisation(model = self.function.function, data = self.data, **self.kwargs)
            optim.optimise()
            recovered = optim.parameters
            fit = [item.get('fun') for item in optim.fit]
            self.output.append( {'fit': fit,
                                      'recover': recovered.copy(),
                                      'original' : parameters.copy()}
            )
        return None
            # for j in range(len(self.data)):
            #     self.function.update()

    def extract(self, key = None):
        if key is None:
            return self.output
        else:
            output = np.zeros((self.iteration, 2, self.population))
            for step in range(self.iteration):
                recovered = [item.get(key) for item in self.output[step].get('recover')]
                original = [item.get(key) for item in self.output[step].get('original')]
                output[step, 0, :] = recovered.copy()
                output[step, 1, :] = original.copy()
            del recovered, original
            return output