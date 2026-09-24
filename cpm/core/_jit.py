"""
Optional compilation with numba.

numba is an optional dependency of cpm (``pip install cpm-toolbox[numba]``): it
lags behind new NumPy releases and does not support PyPy. Modules with
compiled kernels decorate them with `njit` from here, which is numba's `njit`
if numba can be imported and an identity decorator otherwise, so the same
kernels run as plain Python without numba and give the same results.

`kernels` returns such a module either compiled or as plain Python, whatever
is installed. The plain-Python version is a second copy of the module, loaded
from the same source file with compilation turned off, so that kernels calling
other kernels stay uncompiled throughout. That is what `backend="python"` of
the session models uses, for example to debug a model or to check the compiled
version against it.

Compilation can be turned off for a whole process with the environment variable
``CPM_DISABLE_JIT=1`` (or numba's own ``NUMBA_DISABLE_JIT=1``).
"""

import importlib
import importlib.util
import os
import sys

__all__ = ["HAVE_NUMBA", "JIT_ENABLED", "njit", "kernels", "resolve_backend"]

try:
    if os.environ.get("CPM_DISABLE_JIT", "0") not in ("", "0"):
        raise ImportError("compilation disabled with CPM_DISABLE_JIT")
    import numba as _numba

    HAVE_NUMBA = True
except ImportError:  # includes numba installs that reject the installed NumPy
    _numba = None
    HAVE_NUMBA = False

## numba's own switch turns njit into an identity decorator as well
JIT_ENABLED = HAVE_NUMBA and os.environ.get("NUMBA_DISABLE_JIT", "0") in ("", "0")

## set in the namespace of a module copy that is loaded without compilation
PYTHON_FLAG = "__cpm_python__"


def _identity(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda function: function


def njit(*args, **kwargs):
    """numba's `njit` (with ``cache=True`` by default), or an identity decorator without numba."""
    if not HAVE_NUMBA:
        return _identity(*args, **kwargs)
    kwargs.setdefault("cache", True)
    return _numba.njit(*args, **kwargs)


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
        kernels. Without numba, the module itself is plain Python already and is
        returned either way.

    Returns
    -------
    module
    """
    if not python or not JIT_ENABLED:
        return importlib.import_module(name)
    if name not in _PYTHON_MODULES:
        spec = importlib.util.find_spec(name)
        copy_name = f"{name}__python"
        copy_spec = importlib.util.spec_from_file_location(copy_name, spec.origin)
        module = importlib.util.module_from_spec(copy_spec)
        module.__package__ = name.rpartition(".")[0]
        setattr(module, PYTHON_FLAG, True)
        sys.modules[copy_name] = module
        copy_spec.loader.exec_module(module)
        _PYTHON_MODULES[name] = module
    return _PYTHON_MODULES[name]


def resolve_backend(backend):
    """
    The backend a session model runs on.

    Parameters
    ----------
    backend : str
        ``"auto"`` (numba if it is installed and not disabled, plain Python
        otherwise), ``"numba"`` or ``"python"``.

    Returns
    -------
    str
        ``"numba"`` or ``"python"``.
    """
    if backend == "auto":
        return "numba" if JIT_ENABLED else "python"
    if backend == "numba":
        if not HAVE_NUMBA:
            raise ImportError(
                "backend='numba' needs numba, which could not be imported. Install it "
                "with `pip install cpm-toolbox[numba]` (numba supports a NumPy release "
                "some time after it comes out), or use backend='python' or 'auto'."
            )
        return "numba"
    if backend == "python":
        return "python"
    raise ValueError(f"backend must be 'auto', 'numba' or 'python', not {backend!r}.")
