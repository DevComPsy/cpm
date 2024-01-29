import numpy as np
from .. import optimisation
from ..optimisation import utils
from ..optimisation import minimise
from . import strategies
import copy

# The `ParameterRecovery` class is used to recover parameters of a model using
# optimization and strategy techniques.

class ParameterRecovery:
    """
    Class for performing parameter recovery analysis.

    Attributes
    ----------
    model : object
        The model object.
    function : object
        The function associated with the model.
    template : object
        The template parameter.
    optimisation : object
        The optimisation algorithm.
    loss : str
        The type of loss function used for optimisation.
    strategy : object
        The strategy object for generating parameter values.
    parameter_names : list
        A list of parameter names.
    data : list
        A list of dictionaries containing the data.
    iteration : int or float
        The number of iterations for the parameter recovery process.
    population : int
        The number of individuals in the population (how many parameter
        sets should we generate on each iteration).
    kwargs : dict
        Additional keyword arguments.

    Examples
    --------
    >>> from cpm.evaluation import ParameterRecovery

    """

    def __init__(self,
        model=None,
        optimiser=None,
        minimasation="LogLikelihood",
        strategy=None,
        iteration=1000,
        **kwargs
    ):
        """
        Initialize the ParameterRecovery class.

        Parameters
        ----------
        model : object
            The model object.
        optimiser : object
            The optimisation algorithm.
        minimisation : str
            The type of minimisation to be used (e.g., "LogLikelihood").
        strategy : object
            The strategy for generating parameter values.
        iteration : int
            The number of iterations for the parameter recovery process.
        **kwargs : dict
            Additional keyword arguments.

        """
        self.model = copy.deepcopy(model)
        self.function = copy.deepcopy(model.function)
        self.template = self.model.parameters[0]
        self.optimisation = getattr(optimisation, optimiser)
        self.loss = minimasation
        self.strategy = getattr(strategies, strategy)
        self.parameter_names = self.function.parameter_names
        self.data = self.model.data
        self.iteration = iteration
        self.population = len(self.data)
        self.kwargs = kwargs
        self.output = []

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
                template=self.template, population=self.population, **self.kwargs
            )
            self.model.update(parameters=parameters)
            self.model.run()
            self.model.generate()
            data = self.model.generated.copy()
            for i in range(len(self.data)):
                self.data[i]["observed"] = data[i]["observed"]
            optim = self.optimisation(
                model=self.function,
                data=self.data,
                minimisation=self.loss,
                **self.kwargs
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
            output = np.zeros((self.iteration, 2, self.population))
            for step in range(self.iteration):
                recovered = [item.get(key) for item in self.output[step].get("recover")]
                original = [item.get(key) for item in self.output[step].get("original")]
                output[step, 0, :] = recovered.copy()
                output[step, 1, :] = original.copy()
            del recovered, original
            return output
