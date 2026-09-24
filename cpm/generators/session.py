import numpy as np
import pandas as pd
import warnings

from .wrapper import Wrapper
from ..core.data import determine_data_length

__all__ = ["SessionWrapper", "session_data", "session_export"]


def session_data(data):
    """
    Convert the data of one participant into a dictionary of contiguous numpy arrays.

    Parameters
    ----------
    data : pandas.DataFrame or dict
        The data of one participant. If a `pandas.DataFrame`, each row is a trial.
        If a dictionary, each entry holds the values of one variable on all
        trials; its `"ppt"` entry, if any, is left out, as in :class:`Wrapper`.

    Returns
    -------
    dict
        One C-contiguous numpy array per column or entry, with the trials along
        the first axis.
    """
    if isinstance(data, pd.DataFrame):
        return {
            column: np.ascontiguousarray(data[column].to_numpy())
            for column in data.columns
        }
    if isinstance(data, dict):
        return {
            key: np.ascontiguousarray(np.asarray(value))
            for key, value in data.items()
            if key != "ppt"
        }
    raise TypeError(
        f"The data must be a pandas.DataFrame or a dict, not {type(data).__name__}."
    )


def session_export(output, trials):
    """
    Turn the output of a session model into one row per trial.

    Parameters
    ----------
    output : dict
        The output of a session model. Each entry is an array with the trials
        along the first axis, or a scalar, which is repeated on every trial.
    trials : int
        The number of trials.

    Returns
    -------
    pandas.DataFrame
        A table with one row per trial, laid out as :meth:`Wrapper.export` lays
        out the output of a per-trial model: an entry with more than one value per
        trial is split into columns named `<key>_0`, `<key>_1`, and so on, in C
        order, and a `ppt` column is added.
    """
    columns = {}
    for key, value in output.items():
        value = np.asarray(value)
        if value.ndim == 0:
            value = np.repeat(value, trials)
        if value.shape[0] != trials:
            raise ValueError(
                f"The output {key!r} of the session model has {value.shape[0]} rows, "
                f"but the data have {trials} trials. Session models have to return "
                "one row per trial."
            )
        value = value.reshape(trials, -1)
        if value.shape[1] == 1:
            columns[key] = value[:, 0]
        else:
            for i in range(value.shape[1]):
                columns[f"{key}_{i}"] = value[:, i]
    table = pd.DataFrame(columns)
    table["ppt"] = 0
    return table


class SessionWrapper(Wrapper):
    """
    A `Wrapper` for a model function that computes all trials of a participant at once.

    `SessionWrapper` works like :class:`Wrapper <cpm.generators.Wrapper>`, and can
    be used wherever a `Wrapper` can (the optimisers, :class:`Simulator
    <cpm.generators.Simulator>` and `cpm.hierarchical`), but its model function is
    called once per run instead of once per trial. Loops over trials can then be
    written in plain Python on numpy arrays, or compiled with numba, and the cost
    of calling a Python function and copying the parameters on every trial is
    gone. This is typically 10 to 100 times faster than a per-trial `Wrapper`.

    Parameters
    ----------
    model : function
        The model function, which computes the outputs of the model on all trials
        of a participant. See Notes.
    data : pandas.DataFrame or dict
        The data of a single participant (or the environment of a simulation), as
        for :class:`Wrapper`: a `pandas.DataFrame` with one row per trial, or a
        dictionary with one entry per variable, each holding the values on all
        trials.
    parameters : Parameters
        The parameters of the model, including its initial states.
    prepare : function, optional
        A function that turns the data into the form the model function takes. It
        is called once, when the data are set, and not on every run. The default is
        :func:`session_data`, which returns a dictionary of contiguous numpy arrays,
        one per column.

    Notes
    -----
    The model function takes two arguments, `parameters` and `data`. `parameters`
    is the :class:`Parameters <cpm.generators.Parameters>` object of the model, and
    `data` is the output of `prepare`. The model function returns a dictionary of
    outputs, each an array with one row per trial (or a scalar, repeated on every
    trial). If the model is to be fitted, the outputs include

    - 'dependent': the dependent variables for the loss function, with shape
      `(trials,)` or `(trials, k)`.

    After a run, `dependent` has shape `(trials, k)`, as for a `Wrapper`, and each
    output whose name is also a parameter (such as a state) sets that parameter to
    its value on the last trial, the state a per-trial `Wrapper` ends a run with.
    Otherwise the model function should not change `parameters`.

    `export()` returns one row per trial, laid out as the export of a per-trial
    `Wrapper`, see :func:`session_export`.

    Examples
    --------
    A Rescorla-Wagner model of a single cue, written for all trials at once:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from cpm.generators import SessionWrapper, Parameters, Value
    >>> def model(parameters, data):
    ...     alpha = parameters.alpha.value
    ...     value = float(parameters.value.value)
    ...     predictions = np.empty(len(data["reward"]))
    ...     for t, reward in enumerate(data["reward"]):
    ...         predictions[t] = value
    ...         value += alpha * (reward - value)
    ...     return {"prediction": predictions, "dependent": predictions}
    >>> parameters = Parameters(
    ...     alpha=Value(value=0.3, lower=0, upper=1, prior="uniform"),
    ...     value=0.0,
    ... )
    >>> data = pd.DataFrame({"reward": [1, 1, 0, 1], "observed": [0.2, 0.5, 0.6, 0.4]})
    >>> wrapper = SessionWrapper(model=model, data=data, parameters=parameters)
    >>> wrapper.run()
    >>> wrapper.dependent.ravel()
    array([0.   , 0.3  , 0.51 , 0.357])
    """

    def __init__(self, model=None, data=None, parameters=None, prepare=None):
        self.prepare = session_data if prepare is None else prepare
        super().__init__(model=model, data=data, parameters=parameters)
        self.simulation = {}
        self.session = self.prepare(data)

    def run(self):
        """
        Run the model on all trials.

        Returns
        -------
        None
        """
        output = self.model(parameters=self.parameters, data=self.session)
        dependent = output.get("dependent")
        if dependent is not None:
            dependent = np.asarray(dependent, dtype=float)
            if dependent.ndim == 1:
                dependent = dependent.reshape(-1, 1)
            if dependent.shape[0] != self.__len__:
                raise ValueError(
                    f"The dependent variable of the session model has "
                    f"{dependent.shape[0]} rows, but the data have {self.__len__} trials."
                )
            self.dependent = dependent
        self.simulation = output

        ## end the run in the state of the last trial, as a per-trial Wrapper does
        keys = self.parameters.keys()
        final = {}
        for key, value in output.items():
            if key in keys and key != "dependent":
                value = np.asarray(value)
                final[key] = value[-1] if value.ndim > 0 else value
        if final:
            self.parameters.update(**final)

        self.__run__ = True
        return None

    def reset(self, parameters=None, data=None):
        """
        Reset the model.

        Parameters
        ----------
        parameters : dict, array_like, pd.Series or Parameters, optional
            The parameters to reset the model with.
        data : pandas.DataFrame or dict, optional
            New data for the model, which are passed through `prepare` once.

        Notes
        -----
        See :meth:`Wrapper.reset <cpm.generators.Wrapper.reset>`.

        Returns
        -------
        None
        """
        was_run = self.__run__
        super().reset(parameters=parameters, data=data)
        if was_run:
            self.simulation = {}
        if data is not None:
            self.session = self.prepare(data)
        return None

    def export(self):
        """
        Export the trial-level output of the last run.

        Returns
        -------
        pandas.DataFrame
            One row per trial. An output with more than one value per trial is split
            into columns `<key>_0`, `<key>_1`, and so on.
        """
        return session_export(self.simulation, self.__len__)
