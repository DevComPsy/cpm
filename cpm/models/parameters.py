from typing import Any


class Parameters():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__.get(key)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.__dict__

    def export(self):
        return self.__dict__