import numpy as np
from .model import Model

class Wrapper:

    def __init__(self, model, data, params):
        self.model = model
        self.data = data
        self.parameters = params
        self.simulation = []

    def run(self):
        for i in range(len(self.data)):
            # do something
        return None