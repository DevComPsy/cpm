# This file contains the model and generator classes for the model
import numpy as np
form . import utils

class Model:
    def __init__(self, functions = None, policy_maker = None):
        self.act
        self.functions = functions
        self.policy_maker = policy_maker

    def run(self, trial):
        for function in self.functions:
            function()

class Generator:
    def __init__(self, *args, **kwargs):
        super(CLASS_NAME, self).__init__(*args, **kwargs)