import numpy as np
import copy


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
    Value
        A Value object, where each attribute is one of the arguments provided for the function. It support all basic mathematical operations and can be used as a regular float with the parameter value as operand.
    """

    def __init__(self, value=None, prior="normal", lower=0, upper=1):
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
