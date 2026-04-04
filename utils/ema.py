# Exponential Moving Average (EMA) implementation for model parameters

import torch
import copy 

class EMA():
    def __init__(self, model, decay=0.995):

        self.shadow_weigths = copy.deepcopy(model.state_dict())
        self.decay = decay
        self.backup_weights = None

    def update(self, model):

        with torch.no_grad():
            current_state = model.state_dict()
            for key in self.shadow_weigths:
                self.shadow_weigths[key] = self.decay * self.shadow_weigths[key] + (1 - self.decay) * current_state[key]
        
    def apply(self, model):
        self.backup_weights = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow_weigths)

    def restore(self, model):
        model.load_state_dict(self.backup_weights)
        

        