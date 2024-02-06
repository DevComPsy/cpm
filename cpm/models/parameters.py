from typing import Any


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
        self.__dict__.update(kwargs)

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


class ParameterContainer:
    """

    A class representing a parameter with additional details.
    It takes keyword arguments representing the parameters with their values and wraps them into a python object.

    Parameters
    ----------
    value : float
        The value of the parameter.
    prior : string or dict, optional
        The prior distribution of the parameter. If a string, it should be one of the following: 'uniform', 'normal', 'beta', 'gamma', 'exponential'.
        If a dictionary, it should contain the following keys: 'type', 'min', 'max', 'mean', 'sd', 'a', 'b', 'shape', 'rate'.
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


# alpha = ParameterContainer(value=0.5, prior="uniform", lower=0.1, upper=1)
# temperature = ParameterContainer(value=1, prior="uniform", lower=0.1, upper=5)

# params = Parameters(alpha=alpha, temperature=temperature)
