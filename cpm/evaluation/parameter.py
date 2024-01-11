import numpy as np
from .. import optimisation
from ..models import Simulator
from . import strategies

class ParameterRecovery:

    def __init__(self, model, optimisation, sample, data, strategy, *args, **kwargs):
        self.function = model
        self.optimisation = optimisation
        self.sample = sample
        self.data = data
        self.strategy = getattr(strategies, strategy)
    