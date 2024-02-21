from scipy.optimize import differential_evolution
import pandas as pd
import copy
from . import minimise
from . import utils
from ..generators import Simulator, Wrapper


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
        The data used for optimization. An array of dictionaries, where each dictionary contains the data for a single participant.
    loss : function
        The loss function for the objective minimization function.
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    function : [cpm.generators.Wrapper][cpm.generators.Wrapper]
        The model object.
    data : object
        The data used for optimization. An array of dictionaries, where each dictionary contains the data for a single participant.
    loss : function
        The loss function for the objective minimization function.
    kwargs : dict
        Additional keyword arguments.
    fit : list
        List to store the optimization results.
    parameters : list
        List to store the best-fitting parameters as dictionaries.
    participant : object
        The current participant data.
    parameter_names : list
        The names of the model parameters.
    bounds : object
        The parameter bounds for the optimization.

    """

    def __init__(
        self, model=None, bounds=None, data=None, minimisation=LogLikelihood, **kwargs
    ):
        self.function = copy.deepcopy(model)
        self.data = data
        self.loss = minimisation
        self.kwargs = kwargs
        self.fit = []
        self.details = []
        self.parameters = []
        self.participant = data[0]
        if isinstance(model, Wrapper):
            self.parameter_names = self.function.parameter_names
        if isinstance(model, Simulator):
            self.parameter_names = self.function.function.parameter_names
        if hasattr(self.function, "bounds"):
            self.bounds = self.function.bounds
        else:
            self.bounds = bounds
            # raise ValueError("You must define the parameter bounds in the Model object.")
        self.auxiliary = {
            "n": len(self.participant.get("observed")),
            "k": len(self.parameter_names),
        }

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
        evaluated = copy.deepcopy(self.function)
        evaluated.reset(pars)
        evaluated.run()
        predicted = evaluated.policies
        observed = self.participant.get("observed")
        metric = self.loss(predicted, observed, **self.auxiliary)
        del predicted, observed
        if metric == float("inf") or metric == float("-inf") or metric == float("nan"):
            metric = 1e10
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
            # objective = copy.deepcopy(self.minimise)
            result = differential_evolution(self.minimise, self.bounds, **self.kwargs)
            # add the parameters to the list
            self.details.append(result.copy())
            fitted_parameters = utils.ExtractParamsFromFit(
                data=result.x, keys=self.parameter_names
            )
            self.parameters.append(fitted_parameters.copy())
            # add the results to the list
            self.fit.append({"parameters": result.x, "fun": copy.deepcopy(result.fun)})
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

    def export(self):
        """
        Exports the optimization results and fitted parameters as a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing the optimization results and fitted parameters.
        """
        ranged = len(self.parameter_names)
        output = pd.DataFrame()
        for i in range(len(self.fit)):
            current = pd.DataFrame(self.fit[i]["parameters"]).T
            current.columns = self.parameter_names[0 : len(current.columns)]
            current["fun"] = self.fit[i]["fun"]
            output = pd.concat([output, current], axis=0)
        return output
