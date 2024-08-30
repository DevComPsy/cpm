from scipy.optimize import differential_evolution
import pandas as pd
import numpy as np
import copy
import multiprocess as mp


from . import minimise
from ..core.data import decompose, detailed_pandas_compiler, extract_params_from_fit
from ..generators import Simulator, Wrapper
from ..core.optimisers import objective, prepare_data


class DifferentialEvolution:
    """
    Class representing the Differential Evolution optimization algorithm.

    Parameters
    ----------
    model : cpm.generators.Wrapper
        The model to be optimized.
    data : pd.DataFrame, pd.DataFrameGroupBy, list
        The data used for optimization. If a pd.Dataframe, it is grouped by the `ppt_identifier`. If it is a pd.DataFrameGroupby, groups are assumed to be participants. An array of dictionaries, where each dictionary contains the data for a single participant, including information about the experiment and the results too. See Notes for more information.
    minimisation : function
        The loss function for the objective minimization function. Default is `minimise.LogLikelihood.bernoulli`. See the `minimise` module for more information. User-defined loss functions are also supported, but they must conform to the format of currently implemented ones.
    prior: bool
        Whether to include priors in the optimisation. Deafult is 'False'.
    parallel : bool
        Whether to use parallel processing. Default is `False`.
    cl : int
        The number of cores to use for parallel processing. Default is `None`. If `None`, the number of cores is set to 2.
        If `cl` is set to `None` and `parallel` is set to `True`, the number of cores is set to the number of cores available on the machine.
    ppt_identifier : str
        The key in the participant data dictionary that contains the participant identifier. Default is `None`. Returned in the optimization details.
    **kwargs : dict
        Additional keyword arguments. See the [`scipy.optimize.differential_evolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) documentation for what is supported.

    Notes
    -----
    The data parameter must contain all input to the model, including the observed data. The data parameter can be a pandas DataFrame, a pandas DataFrameGroupBy object, or a list of dictionaries. If the data parameter is a pandas DataFrame, it is assumed that the data needs to be grouped by the participant identifier, `ppt_identifier`. If the data parameter is a pandas DataFrameGroupBy object, the groups are assumed to be participants. If the data parameter is a list of dictionaries, each dictionary should contain the data for a single participant, including information about the experiment and the results. The observed data for each participant should be included in the dictionary under the key or column 'observed'. The 'observed' key should correspond, both in format and shape, to the 'dependent' variable calculated by the model Wrapper.
    """

    def __init__(
        self,
        model=None,
        data=None,
        minimisation=minimise.LogLikelihood.bernoulli,
        prior=False,
        parallel=False,
        cl=None,
        ppt_identifier=None,
        display=False,
        **kwargs,
    ):
        if isinstance(model, Simulator):
            raise TypeError(
                "The DifferentialEvolution algorithm is not compatible with the Simulator object."
            )
        self.model = copy.deepcopy(model)
        self.data = data
        self.loss = minimisation
        self.kwargs = kwargs
        self.fit = []
        self.details = []
        self.parameters = []

        self.display = display
        self.ppt_identifier = ppt_identifier
        self.prior = prior

        self.data, self.participants, self.groups, self.__pandas__ = prepare_data(
            data, self.ppt_identifier
        )

        self.parameter_names = self.model.parameters.free()
        bounds = self.model.parameters.bounds()
        bounds = np.asarray(bounds).T
        bounds = list(map(tuple, bounds))
        self.bounds = bounds
        if not self.parameter_names:
            raise ValueError(
                "The model does not contain any free parameters. Please check the model parameters."
            )

        self.__parallel__ = parallel

        if cl is not None:
            self.cl = cl
        if cl is None and parallel:
            self.cl = mp.cpu_count()

    def optimise(self):
        """
        Performs the optimization process.

        Returns
        -------
        None
        """

        loss = self.loss
        model = self.model
        prior = self.prior

        def __task(participant, **args):
            """
            Utility function to wrap fitting the model to each individual for parallel processing and organise the output.
            """

            participant_dc, observed, ppt = decompose(
                participant=participant,
                pandas=self.__pandas__,
                identifier=self.ppt_identifier,
            )

            model.reset(data=participant_dc)

            result = differential_evolution(
                func=objective,
                bounds=self.bounds,
                args=((model, observed, loss, prior)),
                **self.kwargs,
            )

            result.ppt = ppt

            return result

        if self.__parallel__:
            with mp.Pool(self.cl) as pool:
                results = pool.map(__task, self.data)
        else:
            results = list(map(__task, self.data))

        self.details = copy.deepcopy(results)

        for result in results:
            self.parameters.append(
                copy.deepcopy(
                    extract_params_from_fit(data=result.x, keys=self.parameter_names)
                )
            )
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

    def export(self, details=False):
        """
        Exports the optimization results and fitted parameters as a `pandas.DataFrame`.

        Parameters
        ----------
        details : bool
            Whether to include the various metrics related to the optimisation routine in the output.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing the optimization results and fitted parameters. If `details` is `True`, the DataFrame will also include the optimization details.

        Notes
        -----
        The DataFrame will not contain the population and population_energies keys from the optimization details.
        If you want to investigate it, please use the `details` attribute.
        """
        ranged = len(self.parameter_names)
        output = pd.DataFrame()
        for i in range(len(self.fit)):
            current = pd.DataFrame(self.fit[i]["parameters"]).T
            current.columns = self.parameter_names[0 : len(current.columns)]
            current["fun"] = self.fit[i]["fun"]
            output = pd.concat([output, current], axis=0)

        if details:
            metrics = detailed_pandas_compiler(self.details)
            output.reset_index(drop=True, inplace=True)
            metrics.reset_index(drop=True, inplace=True)
            output = pd.concat([output, metrics], axis=1)
        return output
