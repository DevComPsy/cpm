__author__ = "Lenard Dome"
__email__ = "lenarddome@gmail.com"
__version__ = "0.15.8"

"""
This is the main module of the package.
It contains the main class of the package, any kind of setup that has to happen during loading and filtering of warning messages.
Here, we will also define the main function of the package by importing it from subdirectories.
"""

import os
import sys
import logging
import warnings

from .__version__ import __version__
from . import generators
from . import models
from . import applications
from . import optimisation
from . import hierarchical
from . import evaluation
from . import utils

del os, sys, warnings, logging
