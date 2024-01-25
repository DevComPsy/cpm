from scipy.optimize import differential_evolution
import copy
from . import minimise
from . import utils
from ..models import Simulator, Wrapper

class DifferentialEvolution:
    """
    Class representing the Differential Evolution optimization algorithm.

    Parameters
    ----------
    model : object
        The model to be optimized.
    bounds : object
        The parameter bounds for the optimization.
    data : object
        The data used for optimization.
    minimisation : str
        The type of minimization to be performed.
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    function : object
        The model function.
    data : object
        The data used for optimization.
    loss : function
        The loss function for minimization.
    kwargs : dict
        Additional keyword arguments.
    fit : list
        List to store the optimization results.
    parameters : list
        List to store the fitted parameters.
    participant : object
        The current participant data.
    parameter_names : list
        The names of the model parameters.
    bounds : object
        The parameter bounds for the optimization.

    Methods
    -------
    minimise(pars, **args)
        Minimizes the objective function.
    optimise()
        Performs the optimization process.
    reset()
        Resets the optimization results and fitted parameters.
    """
    
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
        """
        The `minimise` function calculates a metric by comparing predicted values with
        observed values.
        
        Parameters
        ----------
        pars
            The `pars` parameter is a dictionary that contains the parameters for the
            function that needs to be minimized.
        
        Returns
        -------
            The metric value is being returned.
        
        """

        self.function.reset()  
        self.function.reset(pars)
        self.function.run()
        predicted = self.function.policies
        observed = self.participant.get("observed")
        metric = self.loss(predicted, observed)
        return metric

    def optimise(self):
        """
        Performs the optimization process.

        Returns:
        - None
        """
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
        """
        Resets the optimization results and fitted parameters.

        Returns:
        - None
        """
        self.fit = []
        self.parameters = []
        return None