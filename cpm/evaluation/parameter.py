# import modules from the cpm package
from .. import optimisation
from ..core import generators
from ..optimisation import minimise
from . import strategies

# import dependencies
import numpy as np
import pandas as pd
import copy
import pickle as pkl


class ParameterRecovery:
    """
    Class for performing parameter recovery analysis.

    Parameters
    ----------
    simulator : Simulator
        The simulator object.
    optimiser : object
        The optimisation algorithm.
    loss : function
        The loss function used for optimisation. It usually takes in the model prediction (output) and the observed data (human responses). We only support minimisation functions.
    strategy : object
        The strategy for generating parameter values.
    iteration : int
        The number of iterations for the parameter recovery process.
    **kwargs : dict
        Additional keyword arguments.

    Examples
    --------
    >>> from cpm.evaluation import ParameterRecovery, strategies
    >>> from cpm.applications import DeltaRule
    >>> from cpm.optimisation import minimise
    >>> from cpm.optimisation import DifferentialEvolution
    >>> model = DeltaRule(parameters, data)
    >>> recover = ParameterRecovery(
            model=model,
            optimiser=DifferentialEvolution,
            loss=minimise.LogLikelihood,
            strategy=strategies.grid,
            iteration=1000
            )
    >>> recover.recover()
    >>> output = recover.extract(key="alpha")
    >>> recover.save(filename="parameter_recovery.pkl")
    >>> del model, recover, output

    """

    def __init__(
        self,
        simulator=None,
        optimiser=None,
        loss=minimise.LogLikelihood.bernoulli,
        strategy=strategies.grid,
        iteration=1000,
        bounds=None,
        variable="dependent",
        **kwargs,
    ):
        """ """
        self.simulator = copy.deepcopy(simulator)
        self.function = copy.deepcopy(simulator.wrapper)
        self.optimisation = optimiser
        self.loss = loss
        self.strategy = strategy
        self.parameter_names = self.function.parameter_names
        self.data = self.simulator.data
        self.iteration = iteration
        self.population = len(self.data)
        self.bounds = bounds
        self.kwargs = kwargs
        self.output = []
        self.variable = variable

        parameters = self.simulator.parameters[0]
        self.template = {
            k: parameters[k] for k in self.parameter_names[0 : len(bounds)]
        }

    def recover(self):
        """
        Recovers parameters using an iterative process.

        This method iteratively updates the parameters of a model, runs the model,
        generates data, and performs optimization to recover the parameters. The
        process is repeated for a specified number of iterations.

        Returns:
            None
        """
        for step in range(self.iteration):
            parameters = self.strategy(
                template=self.template, population=self.population, bounds=self.bounds
            )
            self.simulator.update(parameters=parameters)
            self.simulator.reset()
            self.simulator.run()
            self.simulator.generate(variable=self.variable)
            data = self.simulator.generated.copy()
            for i in range(len(self.data)):
                self.data[i]["observed"] = data[i]["observed"]
            optim = self.optimisation(
                model=self.function,
                data=self.data,
                minimisation=self.loss,
                # bounds=self.bounds,
                **self.kwargs,
            )
            optim.optimise()
            recovered = optim.parameters
            fit = [item.get("fun") for item in optim.fit]
            self.output.append(
                {"fit": fit, "recover": recovered.copy(), "original": parameters.copy()}
            )
            del optim, data, recovered, fit
        return None

    def extract(self, key=None):
        """
        Extract the recovered and original parameter values.

        Parameters
        ----------
        key: str, optional
            The key specifying the parameter value to extract.

        Returns
        -------
        numpy.ndarray
            The extracted parameter values.

        """
        if key is None:
            return self.output
        else:
            output = pd.DataFrame()
            for step in range(self.iteration):
                recovered = [item.get(key) for item in self.output[step].get("recover")]
                original = [item.get(key) for item in self.output[step].get("original")]
                fit = self.output[step].get("fit")

                stepstone = pd.concat(
                    [
                        pd.DataFrame(recovered.copy()),
                        pd.DataFrame(original.copy()),
                        pd.DataFrame(fit.copy()),
                    ],
                    axis=1,
                )

                stepstone.columns = ["recovered", "original", "fit"]
                stepstone["iteration"] = step
                output = pd.concat([output, stepstone], axis=0)
            del recovered, original, stepstone, fit
            return output

    def save(self, filename=None):
        """
        Save the output of the parameter recovery process.

        Parameters
        ----------
        filename : str, optional
            The filename to save the output to, which includes the path.

        Returns
        -------
        None

        """
        if filename is None:
            filename = "parameter_recovery.pkl"
        with open(filename, "wb") as f:
            pkl.dump(self.output, f)
        return None
