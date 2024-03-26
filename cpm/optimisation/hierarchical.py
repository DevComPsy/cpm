import numpy as np
import copy
import pandas as pd
import numpy as np
import copy
from . import minimise
from . import utils
from ..generators import Simulator, Wrapper


class Hierarchical:

    def __init__(
        self,
        simulator=None,
        parameters=None,
        data=None,
        bounds=None,
        loss=None,
        iteration=1000,
        tolerance=1e-6,
        chain=4,
        **kwargs
    ):
        self.simulator = copy.deepcopy(simulator)
        self.data = data
        self.parameters = parameters
        self.bounds = bounds
        self.loss = loss
        self.iteration = iteration
        self.tolerance = tolerance
        self.chain = chain
        self.kwargs = kwargs
        self.fit = []
        self.details = []
        if hasattr(simulator, "bounds"):
            self.bounds = simulator.bounds
        else:
            self.bounds = bounds

    def minimise(self, pars, **args):
        parameters = []
        observations = []
        for i in len(self.data):
            ppt = self.parameters.sample()
            parameters.append(ppt)
            observations.append(self.data[i].get("observed"))
        self.simulator.reset()
        self.simulator.update(parameters)
        self.simulator.run()
        predictions = self.simulator.generate()

        lls = []
        for i in len(self.data):
            metric = self.loss(
                predictions[i].get("observed"), observations[i].get("observed"), **args
            )
            lls.append(metric)
        lls = np.asarray(lls)
        return lls.sum()

    def optimise(self):
        for i in self.chain:
            pass
