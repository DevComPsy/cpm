# This file contains the model and generator classes for the model
import numpy as np
from .zoo import utils

class Model:
    def __init__(self, functions = None):
        self.functions = functions

    def compute(self, **kwargs):
        # TODO: save outputs
        for function in self.functions:
            locator = function.config()['locator']
            if function.config()['type'] == 'decision':
                policy = function(**kwargs)
            else:
                outputs.append(index) = function(**kwargs)
        return x, policy

    def get_config(self):
        return None

    def extract_layers(self, names):
        
        

class Generator:
    def __init__(self, *args, **kwargs):
        super(CLASS_NAME, self).__init__(*args, **kwargs)