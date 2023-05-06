from avalanche.training.plugins import SupervisedPlugin
from avalanche.core import Template
import numpy as np
import torch
class GradientNoise(SupervisedPlugin):

    def __init__(self, eta, gamma):
        super().__init__()
        self.eta = eta
        self.gamma = gamma
    
    def after_backward(self, strategy: Template, *args, **kwargs):
        if not strategy.is_training:
            return
        current_iteration = strategy.clock.train_iterations
        stdev = self.eta / np.power((1+current_iteration), self.gamma)

        for _, layer in strategy.model.named_parameters():
            layer.grad = layer.grad + (torch.randn_like(layer.grad) * stdev)