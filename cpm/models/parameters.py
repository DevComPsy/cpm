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
