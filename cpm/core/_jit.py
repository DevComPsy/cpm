"""
Optional compilation with numba.

numba is an optional dependency of cpm (``pip install cpm-toolbox[numba]``): it
lags behind new NumPy releases and does not support PyPy. Modules with
compiled kernels decorate them with `njit` from here, which is numba's `njit`
if numba can be imported and an identity decorator otherwise, so the same
kernels run as plain Python without numba and give the same results.

`kernels` returns such a module either compiled or as plain Python. The
plain-Python version is a second copy of the module, loaded from the same source
file with compilation turned off, so that kernels calling other kernels stay
uncompiled throughout. The classes in `cpm.models` compute with it, and the
tests use it to check the compiled models against the plain-Python ones.

Compilation can be turned off for a whole process with the environment variable
``CPM_DISABLE_JIT=1`` (or numba's own ``NUMBA_DISABLE_JIT=1``).

numba is imported when it is first needed, that is, when a compiled kernel is
first created, and not when cpm is imported. `HAVE_NUMBA` and `JIT_ENABLED` are
determined then.
"""

import importlib
import importlib.util
import os
import sys
import warnings

__all__ = ["HAVE_NUMBA", "JIT_ENABLED", "njit", "kernels", "resolve_backend"]

## set in the namespace of a module copy that is loaded without compilation
PYTHON_FLAG = "__cpm_python__"

_STATE = {}


def _probe():
    """Import numba, once, and record whether compiled kernels are available."""
    if not _STATE:
        numba = None
        if os.environ.get("CPM_DISABLE_JIT", "0") in ("", "0"):
            try:
                import numba
            except Exception as error:  # such as a numba that rejects the installed NumPy
                if not (isinstance(error, ModuleNotFoundError) and error.name == "numba"):
                    warnings.warn(
                        f"numba is installed but could not be imported ({error!r}), so the "
                        "built-in models run as plain Python, with the same results.",
                        stacklevel=2,
                    )
        _STATE["numba"] = numba
        have = numba is not None
        _STATE["HAVE_NUMBA"] = have
        ## numba's own switch turns njit into an identity decorator as well
        _STATE["JIT_ENABLED"] = have and os.environ.get("NUMBA_DISABLE_JIT", "0") in ("", "0")
    return _STATE


def __getattr__(name):
    if name in ("HAVE_NUMBA", "JIT_ENABLED"):
        return _probe()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _flag(name):
    """`HAVE_NUMBA` or `JIT_ENABLED`; a value set on the module (for example by a test) takes precedence."""
    return globals()[name] if name in globals() else _probe()[name]


def _identity(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda function: function


def njit(*args, **kwargs):
    """numba's `njit` (with ``cache=True`` by default), or an identity decorator without numba."""
    if not _flag("HAVE_NUMBA"):
        return _identity(*args, **kwargs)
    kwargs.setdefault("cache", True)
    numba = _probe()["numba"]
    if not (len(args) == 1 and callable(args[0])):
        return numba.njit(*args, **kwargs)
    try:
        return numba.njit(**kwargs)(args[0])
    except RuntimeError:  # numba has nowhere to write its cache, such as on a read-only install
        return numba.njit(**{**kwargs, "cache": False})(args[0])


def decorator(namespace):
    """The `njit` a module should use: an identity decorator in its plain-Python copy."""
    return _identity if namespace.get(PYTHON_FLAG, False) else njit


_PYTHON_MODULES = {}


def kernels(name, python=False):
    """
    A module of kernels, compiled or as plain Python.

    Parameters
    ----------
    name : str
        The name of the module, for example ``"cpm.models.kernels"``.
    python : bool
        If True, return a copy of the module that is loaded without compiling its
        kernels (and without importing numba). This is what the classes in
        `cpm.models` compute with.

    Returns
    -------
    module
    """
    if not python:
        return importlib.import_module(name)
    if name not in _PYTHON_MODULES:
        spec = importlib.util.find_spec(name)
        copy_name = f"{name}__python"
        copy_spec = importlib.util.spec_from_file_location(copy_name, spec.origin)
        module = importlib.util.module_from_spec(copy_spec)
        module.__package__ = name.rpartition(".")[0]
        setattr(module, PYTHON_FLAG, True)
        sys.modules[copy_name] = module
        ## from the source, since the file does not exist if cpm is imported from a zip
        exec(compile(spec.loader.get_source(name), spec.origin, "exec"), module.__dict__)
        _PYTHON_MODULES[name] = module
    return _PYTHON_MODULES[name]


def resolve_backend(backend):
    """
    The backend a session model runs on.

    Parameters
    ----------
    backend : str
        ``"auto"`` (numba if it is installed and not disabled, plain Python
        otherwise), ``"numba"`` or ``"python"``. The built-in applications use
        ``"auto"``, so that installing numba is all it takes to compile them.

    Returns
    -------
    str
        ``"numba"`` or ``"python"``.
    """
    if backend == "auto":
        return "numba" if _flag("JIT_ENABLED") else "python"
    if backend == "numba":
        if not _flag("HAVE_NUMBA"):
            raise ImportError(
                "Compiling with numba needs numba, which could not be imported. Install "
                "it with `pip install cpm-toolbox[numba]` (numba supports a NumPy release "
                "some time after it comes out)."
            )
        return "numba"
    if backend == "python":
        return "python"
    raise ValueError(f"backend must be 'auto', 'numba' or 'python', not {backend!r}.")
