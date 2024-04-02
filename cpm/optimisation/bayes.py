import numpy as np
import copy
import pandas as pd
import numpy as np
import copy
from . import minimise
from . import utils
from ..generators import Simulator, Wrapper


class EmpiricalBayes:

    def __init__(
        self,
        optimiser=None,
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
        self.optimiser = copy.deepcopy(optimiser)
        self.data = data
        self.parameters = parameters
        # bounds here should include mean and std for all parameters
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
        # TODO: update means and stds for each parameters
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
            # NOTE: palceholders
            self.optimise(
                minimise, self.bounds, self.iteration, self.tolerance, **self.kwargs
            )
            self.fit.append([])
            self.details.append([])
            pass

    def export(self):
        """
        Exports the optimization results and fitted parameters as a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing the optimization results and fitted parameters.
        """
        output = utils.detailed_pandas_compiler(self.details)
        output.reset_index(drop=True, inplace=True)
        return output
