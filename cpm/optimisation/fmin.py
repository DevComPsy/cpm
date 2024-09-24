from ..core.generators import generate_guesses
from ..core.optimisers import objective, numerical_hessian, prepare_data
from ..core.data import detailed_pandas_compiler, decompose
from ..core.parallel import detect_cores, execute_parallel
from ..generators import Simulator, Wrapper

from scipy.optimize import fmin, fmin_l_bfgs_b

import numpy as np
import pandas as pd
import multiprocess as mp
import copy

__all__ = ["Fmin", "FminBound"]


class Fmin:
    """
    Class representing the Fmin search (unbounded) optimization algorithm using a downhill simplex.

    Parameters
    ----------
    model : cpm.generators.Wrapper
        The model to be optimized.
    data : pd.DataFrame, pd.DataFrameGroupBy, list
        The data used for optimization. If a pd.Dataframe, it is grouped by the `ppt_identifier`. If it is a pd.DataFrameGroupby, groups are assumed to be participants. An array of dictionaries, where each dictionary contains the data for a single participant, including information about the experiment and the results too. See Notes for more information.
    minimisation : function
        The loss function for the objective minimization function. See the `minimise` module for more information. User-defined loss functions are also supported.
    prior: bool
        Whether to include the prior in the optimization. Default is `False`.
    number_of_starts : int
        The number of random initialisations for the optimization. Default is `1`.
    initial_guess : list or array-like
        The initial guess for the optimization. Default is `None`. If `number_of_starts` is set, and the `initial_guess` parameter is 'None', the initial guesses are randomly generated from a uniform distribution.
    parallel : bool
        Whether to use parallel processing. Default is `False`.
    cl : int
        The number of cores to use for parallel processing. Default is `None`. If `None`, the number of cores is set to 2.
        If `cl` is set to `None` and `parallel` is set to `True`, the number of cores is set to the number of cores available on the machine.
    libraries : list, optional
        The libraries to import for parallel processing for `ipyparallel` with the IPython kernel. Default is `["numpy", "pandas"]`.
    ppt_identifier : str
        The key in the participant data dictionary that contains the participant identifier. Default is `None`. Returned in the optimization details.
    **kwargs : dict
        Additional keyword arguments. See the [`scipy.optimize.fmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html) documentation for what is supported.


    Notes
    -----
    The data parameter must contain all input to the model, including the observed data. The data parameter can be a pandas DataFrame, a pandas DataFrameGroupBy object, or a list of dictionaries. If the data parameter is a pandas DataFrame, it is assumed that the data needs to be grouped by the participant identifier, `ppt_identifier`. If the data parameter is a pandas DataFrameGroupBy object, the groups are assumed to be participants. If the data parameter is a list of dictionaries, each dictionary should contain the data for a single participant, including information about the experiment and the results. The observed data for each participant should be included in the dictionary under the key or column 'observed'. The 'observed' key should correspond, both in format and shape, to the 'dependent' variable calculated by the model Wrapper.

    The optimization process is repeated `number_of_starts` times, and only the best-fitting output from the best guess is stored.
    """

    def __init__(
        self,
        model=None,
        data=None,
        initial_guess=None,
        minimisation=None,
        cl=None,
        parallel=False,
        libraries=["numpy", "pandas"],
        prior=False,
        number_of_starts=1,
        ppt_identifier=None,
        display=False,
        **kwargs,
    ):
        self.model = copy.deepcopy(model)
        self.data = data
        self.ppt_identifier = ppt_identifier
        self.data, self.participants, self.groups, self.__pandas__ = prepare_data(
            data, self.ppt_identifier
        )

        self.loss = minimisation
        self.prior = prior
        self.kwargs = kwargs
        self.display = display
        self.prior = prior

        self.fit = []
        self.details = []
        self.parameters = []

        if isinstance(model, Wrapper):
            self.parameter_names = self.model.parameters.free()
        if not self.parameter_names:
            raise ValueError(
                "The model does not contain any free parameters. Please check the model parameters."
            )
        if isinstance(model, Simulator):
            raise TypeError(
                "The Fmin algorithm is not compatible with the Simulator object."
            )

        self.initial_guess = generate_guesses(
            bounds=self.model.parameters.bounds(),
            number_of_starts=number_of_starts,
            guesses=initial_guess,
            shape=(number_of_starts, len(self.parameter_names)),
        )

        self.__parallel__ = parallel
        self.__current_guess__ = self.initial_guess[0]
        self.__libraries__ = libraries

        if cl is not None:
            self.cl = cl
        if cl is None and parallel:
            self.cl = detect_cores()

    def optimise(self):
        """
        Performs the optimization process.

        Returns:
        - None
        """

        def __unpack(x, id=None):
            keys = ["xopt", "fopt", "iter", "funcalls", "warnflag", "hessian"]
            if id is not None:
                keys.append(id)
            out = {}
            for i in range(len(keys)):
                out[keys[i]] = x[i]
            out["fun"] = out.pop("fopt")
            return out

        def __task(participant, **args):

            participant_dc, observed, ppt = decompose(
                participant=participant,
                pandas=self.__pandas__,
                identifier=self.ppt_identifier,
            )

            model.reset(data=participant_dc)

            result = fmin(
                objective,
                x0=self.__current_guess__,
                args=(model, observed, loss, prior),
                disp=self.display,
                **self.kwargs,
                full_output=True,
            )

            def f(x):
                return objective(x, model, observed, loss, prior)

            hessian = numerical_hessian(func=f, params=result[0] + 1e-3)
            result = (*result, hessian)

            # if participant data contains identifiers, return the identifiers too
            result = (*result, ppt)
            return result

        def __extract_nll(result):
            output = np.zeros(len(result))
            for i in range(len(result)):
                output[i] = result[i][1]
            return output.copy()

        loss = self.loss
        model = self.model
        prior = self.prior

        for i in range(len(self.initial_guess)):
            print(
                f"Starting optimization {i+1}/{len(self.initial_guess)} from {self.initial_guess[i]}"
            )
            self.__current_guess__ = self.initial_guess[i]
            if self.__parallel__:
                results = execute_parallel(
                    job=__task,
                    data=self.data,
                    method=None,
                    cl=self.cl,
                    libraries=self.__libraries__,
                )
            else:
                results = list(map(__task, self.data))

            ## extract the negative log likelihoods for each ppt
            if i == 0:
                old_nll = __extract_nll(results)
                self.details = copy.deepcopy(results)
                parameters = {}
                for result in results:
                    for i in range(len(self.parameter_names)):
                        parameters[self.parameter_names[i]] = copy.deepcopy(
                            result[0][i]
                        )
                    self.parameters.append(copy.deepcopy(parameters))
                    self.fit.append(
                        __unpack(copy.deepcopy(result), id=self.ppt_identifier)
                    )
            else:
                nll = __extract_nll(results)
                # check if ppt fit is better than the previous fit
                indices = np.where(nll < old_nll)[0]
                for ppt in indices:
                    self.details[ppt] = copy.deepcopy(results[ppt])
                    for i in range(len(self.parameter_names)):
                        self.parameters[ppt][self.parameter_names[i]] = copy.deepcopy(
                            results[ppt][0][i]
                        )
                    self.fit[ppt] = __unpack(
                        copy.deepcopy(results[ppt]), id=self.ppt_identifier
                    )

        return None

    def reset(self, initial_guess=True):
        """
        Resets the optimization results and fitted parameters.

        Parameters
        ----------
        initial_guess : bool, optional
            Whether to reset the initial guess (generates a new set of random numbers within parameter bounds). Default is `True`.

        Returns
        -------
        None
        """
        self.fit = []
        self.details = []
        self.parameters = []
        if initial_guess:
            self.initial_guess = generate_guesses(
                bounds=self.model.parameters.bounds(),
                number_of_starts=self.initial_guess.shape[0],
                guesses=None,
                shape=self.initial_guess.shape,
            )
        return None

    def export(self):
        """
        Exports the optimization results and fitted parameters as a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing the optimization results and fitted parameters.
        """
        output = detailed_pandas_compiler(self.fit)
        output.reset_index(drop=True, inplace=True)
        return output


class FminBound:
    """
    Class representing the Fmin search (bounded) optimization algorithm using the L-BFGS-B method.

    Parameters
    ----------
    model : cpm.generators.Wrapper
        The model to be optimized.
    data : pd.DataFrame, pd.DataFrameGroupBy, list
        The data used for optimization. If a pd.Dataframe, it is grouped by the `ppt_identifier`. If it is a pd.DataFrameGroupby, groups are assumed to be participants. An array of dictionaries, where each dictionary contains the data for a single participant, including information about the experiment and the results too. See Notes for more information.
    minimisation : function
        The loss function for the objective minimization function. See the `minimise` module for more information. User-defined loss functions are also supported.
    prior: bool
        Whether to include the prior in the optimization. Default is `False`.
    number_of_starts : int
        The number of random initialisations for the optimization. Default is `1`.
    initial_guess : list or array-like
        The initial guess for the optimization. Default is `None`. If `number_of_starts` is set, and the `initial_guess` parameter is 'None', the initial guesses are randomly generated from a uniform distribution.
    parallel : bool
        Whether to use parallel processing. Default is `False`.
    cl : int
        The number of cores to use for parallel processing. Default is `None`. If `None`, the number of cores is set to 2.
        If `cl` is set to `None` and `parallel` is set to `True`, the number of cores is set to the number of cores available on the machine.
    libraries : list, optional
        The libraries to import for parallel processing for `ipyparallel` with the IPython kernel. Default is `["numpy", "pandas"]`.
    ppt_identifier : str
        The key in the participant data dictionary that contains the participant identifier. Default is `None`. Returned in the optimization details.
    **kwargs : dict
        Additional keyword arguments. See the [`scipy.optimize.fmin_l_bfgs_b`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html) documentation for what is supported.


    Notes
    -----
    The data parameter must contain all input to the model, including the observed data. The data parameter can be a pandas DataFrame, a pandas DataFrameGroupBy object, or a list of dictionaries. If the data parameter is a pandas DataFrame, it is assumed that the data needs to be grouped by the participant identifier, `ppt_identifier`. If the data parameter is a pandas DataFrameGroupBy object, the groups are assumed to be participants. If the data parameter is a list of dictionaries, each dictionary should contain the data for a single participant, including information about the experiment and the results. The observed data for each participant should be included in the dictionary under the key or column 'observed'. The 'observed' key should correspond, both in format and shape, to the 'dependent' variable calculated by the model Wrapper.

    The optimization process is repeated `number_of_starts` times, and only the best-fitting output from the best guess is stored.
    """

    def __init__(
        self,
        model=None,
        data=None,
        initial_guess=None,
        number_of_starts=1,
        minimisation=None,
        cl=None,
        parallel=False,
        libraries=["numpy", "pandas"],
        prior=False,
        ppt_identifier=None,
        display=False,
        **kwargs,
    ):
        self.model = copy.deepcopy(model)
        self.data = data
        self.ppt_identifier = ppt_identifier
        self.data, self.participants, self.groups, self.__pandas__ = prepare_data(
            data, self.ppt_identifier
        )

        self.loss = minimisation
        self.prior = prior
        self.kwargs = kwargs
        self.display = display
        self.prior = prior

        self.fit = []
        self.details = []
        self.parameters = []

        if isinstance(model, Wrapper):
            self.parameter_names = self.model.parameters.free()
        if not self.parameter_names:
            raise ValueError(
                "The model does not contain any free parameters. Please check the model parameters."
            )
        if isinstance(model, Simulator):
            raise ValueError(
                "The Fmin algorithm is not compatible with the Simulator object."
            )

        self.initial_guess = generate_guesses(
            bounds=self.model.parameters.bounds(),
            number_of_starts=number_of_starts,
            guesses=initial_guess,
            shape=(number_of_starts, len(self.parameter_names)),
        )

        self.__parallel__ = parallel
        self.__current_guess__ = self.initial_guess[0]
        self.__libraries__ = libraries

        if cl is not None:
            self.cl = cl
        if cl is None and parallel:
            self.cl = detect_cores()

    def optimise(self, display=True):
        """
        Performs the optimization process.

        Returns:
        - None
        """

        def __unpack(x, id=None):
            keys = ["x", "f", "grad", "task", "funcalls", "nit", "warnflag", "hessian"]
            if id is not None:
                keys.append(id)
            out = {}
            for i in range(len(keys)):
                out[keys[i]] = x[i]
            out["fun"] = out.pop("f")
            return out

        bounds = self.model.parameters.bounds()
        bounds = np.asarray(bounds).T
        bounds = list(map(tuple, bounds))
        loss = self.loss
        model = self.model
        prior = self.prior

        def __task(participant, **args):

            participant_dc, observed, ppt = decompose(
                participant=participant,
                pandas=self.__pandas__,
                identifier=self.ppt_identifier,
            )

            model.reset(data=participant_dc)

            result = fmin_l_bfgs_b(
                objective,
                x0=self.__current_guess__,
                bounds=bounds,
                args=(model, observed, loss, prior),
                disp=self.display,
                **self.kwargs,
            )

            def f(x):
                return objective(x, model, observed, loss, prior)

            hessian = numerical_hessian(func=f, params=result[0] + 1e-3)

            result = (*result[0:2], *tuple(list(result[2].values())), hessian)

            result = (*result, ppt)
            return result

        def __extract_nll(result):
            output = np.zeros(len(result))
            for i in range(len(result)):
                output[i] = result[i][1]
            return output.copy()

        for i in range(len(self.initial_guess)):
            print(
                f"Starting optimization {i+1}/{len(self.initial_guess)} from {self.initial_guess[i]}"
            )
            self.__current_guess__ = self.initial_guess[i]
            if self.__parallel__:
                results = execute_parallel(
                    job=__task,
                    data=self.data,
                    method=None,
                    cl=self.cl,
                    libraries=self.__libraries__,
                )
            else:
                results = list(map(__task, self.data))

            ## extract the negative log likelihoods for each ppt
            if i == 0:
                old_nll = __extract_nll(results)
                self.details = copy.deepcopy(results)
                parameters = {}
                for result in results:
                    for i in range(len(self.parameter_names)):
                        parameters[self.parameter_names[i]] = copy.deepcopy(
                            result[0][i]
                        )
                    self.parameters.append(copy.deepcopy(parameters))
                    self.fit.append(
                        __unpack(copy.deepcopy(result), id=self.ppt_identifier)
                    )
            else:
                nll = __extract_nll(results)
                # check if ppt fit is better than the previous fit
                indices = np.where(nll < old_nll)[0]
                for ppt in indices:
                    self.details[ppt] = copy.deepcopy(results[ppt])
                    for i in range(len(self.parameter_names)):
                        self.parameters[ppt][self.parameter_names[i]] = copy.deepcopy(
                            results[ppt][0][i]
                        )
                    self.fit[ppt] = __unpack(
                        copy.deepcopy(results[ppt]), id=self.ppt_identifier
                    )

        return None

    def reset(self, initial_guess=True):
        """
        Resets the optimization results and fitted parameters.

        Parameters
        ----------
        initial_guess : bool, optional
            Whether to reset the initial guess (generates a new set of random numbers within parameter bounds). Default is `True`.

        Returns
        -------
        None
        """
        self.fit = []
        self.details = []
        self.parameters = []
        if initial_guess:
            self.initial_guess = generate_guesses(
                bounds=self.model.parameters.bounds(),
                number_of_starts=self.initial_guess.shape[0],
                guesses=None,
                shape=self.initial_guess.shape,
            )
        return None

    def export(self):
        """
        Exports the optimization results and fitted parameters as a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame containing the optimization results and fitted parameters.
        """
        output = detailed_pandas_compiler(self.fit)
        output.reset_index(drop=True, inplace=True)
        return output
