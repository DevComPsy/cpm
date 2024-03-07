import numpy as np
import copy
from scipy.stats import truncnorm, truncexpon, uniform, beta, gamma


class Parameters:
    """
    A class representing a set of parameters.
    It takes keyword arguments representing the parameters with their values and wraps them into a python object.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments representing the parameters.

    Returns
    -------
    Parameters
        A Parameters object, where each attributes is one of the keyword arguments
        provided for the function.

    Examples
    --------
    >>> from cpm.generators import Parameters
    >>> parameters = Parameters(a=0.5, b=0.5, c=0.5)
    >>> parameters['a']
    0.1
    >>> parameters.a
    0.1
    >>> parameters()
    {'a': 0.1, 'b': 0.2, 'c': 0.5}
    >>> parameters.prior()
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, Value(value))
        # self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return self.__dict__.get(key)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __call__(self):
        return self.__dict__

    def export(self):
        return self.__dict__

    def __copy__(self):
        return Parameters(**self.__dict__)

    def __keys__(self):
        return self.__dict__.keys()

    def __len__(self):
        return 0

    def update(self, **kwargs):
        """
        Update the parameters with new values.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments representing the parameters.

        """
        for key, value in kwargs.items():
            setattr(self, key, Value(value))

    def keys(self):
        """
        Return a list of all the keys in the parameters dictionary.

        Returns
        -------
        keys : list
            A list of all the keys in the parameters dictionary.
        """
        return self.__dict__.keys()

    def prior(self, log=False):
        """
        Return the prior distribution of the parameters.

        Returns
        -------
        The probability of the parameter set under the prior distribution for each parameter.
        If `log` is True, the log probability is returned.
        """
        prior = 1
        for key, value in self.__dict__.items():
            prior *= value.prior()
        if log:
            prior = np.log(prior)
        return prior


class Value:
    """

    The `Value` class is a wrapper around a float value, with additional details such as the prior distribution, lower and upper bounds. It supports all basic mathematical operations and can be used as a regular float with the parameter value as operand.


    Parameters
    ----------
    value : float
        The value of the parameter.
    prior : string or object, optional
        If a string, it should be one of continuous distributions from `scipy.stats`.
        See the [scipy documentation](https://docs.scipy.org/doc/scipy/reference/stats.html) for more details.
        The default is 'normal'.
        If an object, it should be or contain a callable function representing the prior distribution of the parameter.
        See Notes for more details.
    lower : float, optional
        The lower bound of the parameter.
    upper : float, optional
        The upper bound of the parameter.

    Attributes
    ----------
    value : float
        The value of the parameter.
    prior : function, optional

    lower : float, optional
        The lower bound of the parameter.
    upper : float, optional
        The upper bound of the parameter.

    Notes
    -----
    We currently implement the following continuous distributions from `scipy.stats`:

    - 'uniform'
    - 'normal'
    - 'beta'
    - 'gamma'
    - 'exponential'

    Because these distributions are inherited from `scipy.stats`, see the [scipy documentation](https://docs.scipy.org/doc/scipy/reference/stats.html) for more details on how to update variables of the distribution.

    Returns
    -------
    Value
        A Value object, where each attribute is one of the arguments provided for the function. It support all basic mathematical operations and can be used as a regular float with the parameter value as operand.
    """

    def __init__(self, value=None, prior="normal", lower=0, upper=1):
        self.value = value

        self.lower = lower
        self.upper = upper

        __sd__ = np.mean([upper, lower]) / 2

        # set the prior distribution
        self.__builtin__ = True
        if prior == "uniform":
            self.priorf = uniform(loc=lower, scale=upper)
        elif prior == "normal":
            self.priorf = truncnorm(
                loc=np.mean([lower, upper]), scale=__sd__, a=lower, b=upper
            )
        elif prior == "beta":
            self.priorf = beta(a=1, b=1, loc=lower, scale=upper - lower)
        elif prior == "gamma":
            self.priorf = gamma(a=1, loc=lower, scale=upper - lower)
        elif prior == "exponential":
            self.priorf = truncexpon(b=upper - lower, loc=lower, scale=__sd__)
        elif isinstance(prior, function):
            self.__builtin__ = False
            self.priorf = prior

    def __repr__(self):
        return str(self.value)

    def __getitem__(self, key):
        return self.__dict__.get(key)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __update__(self, key, value):
        self.__dict__[key] = value

    def __call__(self):
        return self.__dict__

    def export(self):
        return self.__dict__

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):  # other * self
        return self.value * other

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):  # other + self
        return self.value + other

    def __iadd__(self, other):
        return self.value + other

    def __sub__(self, other):  # self - other
        return self.value + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self.value)

    def __truediv__(self, other):
        return self.value / other

    def __floordiv__(self, other):
        return self.value // other

    def __mod__(self, other):
        return self.value % other

    def __pow__(self, other):
        return self.value**other

    def __rpow__(self, other):
        return other**self.value

    def __neg__(self):  # -self
        return self.value * -1

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __lt__(self, other):
        return self.value < other

    def __neg__(self):  # -self
        return self.value * -1

    def __truediv__(self, other):  # self / other
        return self.value * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self.value**-1

    def __copy__(self):
        return Value(**self.__dict__)

    def __array__(self) -> np.ndarray:
        return np.array(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __len__(self):
        return len(self.value)

    def __shape__(self):
        return np.shape(self.value)

    def __copy__(self):
        return Value(**self.__dict__)

    def __deepcopy__(self, memo):
        return Value(**copy.deepcopy(self.__dict__, memo))

    def copy(self):
        """
        Return a copy of the parameter.

        Returns
        -------
        Value
            A copy of the parameter.
        """
        return Value(**self.__dict__)

    def fill(self, value):
        """
        Replace the value of the parameter with a new value.

        Parameters
        ----------
        value : float or array_like
            The new value of the parameter.

        Notes
        -----
        If the parameter is an array, it should be a list of values. If the parameter is an array, and the new value is a single value, it will be broadcasted to the shape of the array.
        """
        self = Value(value)

    def prior(self, log=False):
        """
        Return the prior distribution of the parameter.

        Returns
        -------
        The probability of the parameter value under the prior distribution.
        If `log` is True, the log probability is returned.
        """
        if log:
            return self.priorf.logpdf(self.value)
        else:
            return self.priorf.pdf(self.value)

    def prior_update(self, **kwargs):
        """
        Set the prior distribution of the parameter.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments representing the prior distribution of the parameter.
            It should contain the necessary arguments for the prior distribution function.
            See the [scipy documentation](https://docs.scipy.org/doc/scipy/reference/stats.html) for more details on what is allowed for each option with the built-in priors.

        Raises
        ------
        ValueError
            If the prior distribution of this parameter is a user-supplied function, it cannot be modified with the `prior` method.

        """
        if self.__builtin__ and kwargs is not None:
            self.priorf.kws.update(**kwargs)
        if not self.__builtin__ and kwargs is not None:
            raise ValueError(
                "The prior distribution of this parameter is a user-supplied function. It cannot be modified with the `prior` method."
            )
