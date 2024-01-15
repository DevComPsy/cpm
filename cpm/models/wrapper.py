import numpy as np
from .model import Model
from .zoo.utils import Nominal

class Wrapper:

    def __init__(self, model, data, params):
        self.model = model
        self.data = data
        self.parameters = params
        self.simulation = []
        self.outcomes = self.data['outcomes']
        self.outcomes = self.data['outcomes']

    def run(self):
        
        for i in range(len(self.data)):
            # do something
            trial = {}
            trial['input'] = Nominal(stimuli = self.training[i], bits = self.bits)
            feedback = Nominal(stimuli = [self.feedback[i]], bits = self.outcomes)
            
        return None

# arguments = {
#     'alpha' : 0.1,
#     'temperature': 1,
#     'weights' : np.array([[0.5, 0], [0, 0.5]]),
#     'input' : np.array([1, 1]),
#     'teacher' : np.array([1, 0]),
#     'attention' : np.array([1, 0]),
#     'misc' : np.array([1, 0])
#     }