from typing import Any
import numpy as np


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

    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, Value(value))
        # self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)

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


class Value:
    """

    A class representing a parameter with additional details in a hierarchical structure.

    Parameters
    ----------
    value : float
        The value of the parameter.
    prior : string or dict, optional
        The prior distribution of the parameter. If a string, it should be one of the following: 'uniform', 'normal', 'beta', 'gamma', 'exponential'.
        If a dictionary, it should contain the following keys: 'type', 'min', 'max', 'mean', 'sd', 'a', 'b', 'shape', 'rate'.
        The default is 'normal'.
    lower : float, optional
        The lower bound of the parameter.
    upper : float, optional
        The upper bound of the parameter.

    Attributes
    ----------
    value : float
        The value of the parameter.
    prior : string or dict, optional
        The prior distribution of the parameter. If a string, it should be one of the following: 'uniform', 'normal', 'beta', 'gamma', 'exponential'.
        If a dictionary, it should contain the following keys: 'type', 'min', 'max', 'mean', 'sd', 'a', 'b', 'shape', 'rate'.
        The default is 'normal'.
    lower : float, optional
        The lower bound of the parameter.
    upper : float, optional
        The upper bound of the parameter.

    Returns
    -------
    ParameterContainer
        A ParameterContainer object, where each attribute is one of the arguments provided for the function. It support all basic mathematical operations and can be used as a regular float.
    """

    # TODO: Add support for different types of priors

    def __init__(self, value=None, prior="normal", lower=None, upper=None):
        self.value = value
        self.prior = prior
        self.lower = lower
        self.upper = upper

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
        return ParameterContainer(**self.__dict__)

    def __array__(self) -> np.ndarray:
        return np.array(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def fill(self, value):
        if hasattr(value, "__iter__"):
            self.value = np.array(value)
        elif hasattr(self.value, "__iter__"):
            self.value.fill(value)
        else:
            self.value = value

    def __len__(self):
        return len(self.value)

    def __shape__(self):
        return np.shape(self.value)
